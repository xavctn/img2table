# coding: utf-8
from typing import Tuple, Optional, List

import cv2
import numpy as np
import polars as pl

from img2table.tables.objects.cell import Cell


def compute_char_length(img: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Compute average character length based on connected components analysis
    :param img: image array
    :return: tuple with average character length and connected components array
    """
    # Thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components
    _, _, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # Remove connected components with less than 15 pixels
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 15
    stats = stats[mask_pixels]

    if len(stats) == 0:
        return None, None

    # Create mask to remove connected components corresponding to the complete image
    mask_height = img.shape[0] > stats[:, cv2.CC_STAT_HEIGHT]
    mask_width = img.shape[1] > stats[:, cv2.CC_STAT_WIDTH]
    mask_img = mask_width & mask_height

    # Compute median width and height
    median_width = np.median(stats[:, cv2.CC_STAT_WIDTH])
    median_height = np.median(stats[:, cv2.CC_STAT_HEIGHT])

    # Compute bbox area bounds
    upper_bound = 4 * median_width * median_height
    lower_bound = 0.25 * median_width * median_height

    # Filter connected components according to their area
    mask_lower_area = lower_bound < stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_area = upper_bound > stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_area = mask_lower_area & mask_upper_area

    # Filter connected components from mask
    stats = stats[mask_img & mask_area]

    if len(stats) > 0:
        # Compute average character length
        char_length = np.mean(stats[:, cv2.CC_STAT_WIDTH])

        return char_length, stats
    else:
        return None, None


def compute_median_line_sep(img: np.ndarray, cc: np.ndarray,
                            char_length: float) -> Tuple[Optional[float], Optional[List[Cell]]]:
    """
    Compute median separation between lines
    :param img: image array
    :param cc: connected components array
    :param char_length: average character length
    :return: median separation between lines
    """
    # Create image from connected components
    black_img = np.zeros(img.shape, np.uint8)
    for c in cc:
        cv2.rectangle(black_img,
                      (c[cv2.CC_STAT_LEFT], c[cv2.CC_STAT_TOP]),
                      (c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH], c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT]),
                      (255, 255, 255), -1)

    # Dilate image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (round(char_length), 1))
    dilate = cv2.dilate(black_img, kernel, iterations=1)

    # Find and map contours
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = list()
    for idx, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        contours.append({"id": idx, "x1": x, "y1": y, "x2": x + w, "y2": y + h})

    # Create contours dataframe
    df_contours = pl.LazyFrame(data=contours)

    # Cross join to get corresponding contours and filter on contours that corresponds horizontally
    df_h_cnts = (df_contours.join(df_contours, how='cross')
                 .filter(pl.col('id') != pl.col('id_right'))
                 .filter(pl.min([pl.col('x2'), pl.col('x2_right')])
                         - pl.max([pl.col('x1'), pl.col('x1_right')]) > 0)
                 )

    # Get contour which is directly below
    df_cnts_below = (df_h_cnts.filter(pl.col('y1') < pl.col('y1_right'))
                     .sort(['id', 'y1_right'])
                     .with_columns(pl.lit(1).alias('ones'))
                     .with_columns(pl.col('ones').cumsum().over(["id"]).alias('rk'))
                     .filter(pl.col('rk') == 1)
                     )

    if df_cnts_below.collect(streaming=True).height == 0:
        return None, [Cell(x1=c.get('x1'), y1=c.get('y1'), x2=c.get('x2'), y2=c.get('y2')) for c in contours]

    # Compute median vertical distance between contours
    median_v_dist = (df_cnts_below.with_columns(((pl.col('y1_right') + pl.col('y2_right')
                                                   - pl.col('y1') - pl.col('y2')) / 2).abs().alias('y_diff'))
                     .select(pl.median('y_diff'))
                     .collect(streaming=True)
                     .to_dicts()
                     .pop()
                     .get('y_diff')
                     )

    return median_v_dist, [Cell(x1=c.get('x1'), y1=c.get('y1'), x2=c.get('x2'), y2=c.get('y2')) for c in contours]


def compute_img_metrics(img: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[List[Cell]]]:
    """
    Compute metrics from image
    :param img: image array
    :return: average character length, median line separation and image contours
    """
    # Compute average character length based on connected components analysis
    char_length, cc_array = compute_char_length(img=img)

    if char_length is None:
        return None, None, None

    # Compute median separation between lines
    median_line_sep, contours = compute_median_line_sep(img=img, cc=cc_array, char_length=char_length)

    return char_length, median_line_sep, contours
