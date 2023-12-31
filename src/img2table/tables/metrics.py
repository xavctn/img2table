# coding: utf-8
from typing import Tuple, Optional, List

import cv2
import numpy as np
import polars as pl
from numba import njit, prange

from img2table.tables.objects.cell import Cell


@njit("List(int64)(int32[:,:],int32[:,:])", fastmath=True, cache=True, parallel=False)
def remove_dots(cc_labels: np.ndarray, stats: np.ndarray) -> List[int]:
    """
    Remove dots from connected components
    :param cc_labels: connected components' label array
    :param stats: connected components' stats array
    :return: list of non-dot connected components' indexes
    """
    cc_to_keep = list()

    for idx in prange(len(stats)):
        if idx == 0:
            continue

        x, y, w, h, area = stats[idx][:]

        # If it is not squared, continue
        if max(w, h) / min(w, h) >= 1.5:
            cc_to_keep.append(idx)
            continue

        # Check number of inner pixels
        inner_pixels = 0
        for row in prange(y, y + h):
            prev_position = -1
            for col in range(x, x + w):
                value = cc_labels[row][col]
                if value == idx:
                    if prev_position >= 0:
                        inner_pixels += col - prev_position - 1
                    prev_position = col

        for col in prange(x, x + w):
            prev_position = -1
            for row in range(y, y + h):
                value = cc_labels[row][col]
                if value == idx:
                    if prev_position >= 0:
                        inner_pixels += row - prev_position - 1
                    prev_position = row

        if not inner_pixels / (2 * area) <= 0.05:
            cc_to_keep.append(idx)

    return cc_to_keep


def compute_char_length(img: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Compute average character length based on connected components' analysis
    :param img: image array
    :return: tuple with average character length and connected components array
    """
    # Thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components
    _, cc_labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # Remove dots
    cc_to_keep = remove_dots(cc_labels=cc_labels, stats=stats)
    stats = stats[cc_to_keep, :]

    # Remove connected components with less than 10 pixels
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 10
    stats = stats[mask_pixels]

    if len(stats) == 0:
        return None, None

    # Filter components based on aspect ratio
    mask_ar = (np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
               / np.minimum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])) <= 2

    # Filter components based on fill ratio
    mask_fill = stats[:, cv2.CC_STAT_AREA] / (stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]) > 0.08

    stats = stats[mask_ar & mask_fill]

    if len(stats) == 0:
        return None, None

    # Compute median width and height
    median_width = np.median(stats[:, cv2.CC_STAT_WIDTH])
    median_height = np.median(stats[:, cv2.CC_STAT_HEIGHT])

    # Compute bbox area bounds
    upper_bound = 4 * median_width * median_height
    lower_bound = 0.25 * median_width * median_height

    # Filter connected components according to their area
    mask_lower_area = lower_bound <= stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_area = upper_bound >= stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_area = mask_lower_area & mask_upper_area

    # Filter connected components from mask
    stats = stats[mask_area]

    if len(stats) > 0:
        # Compute average character length
        char_length = np.mean(stats[:, cv2.CC_STAT_WIDTH])

        return char_length, stats
    else:
        return None, None


def recompute_contours(cells_cc: List[Cell], df_contours: pl.LazyFrame) -> List[Cell]:
    """
    Recompute contours identified with original cells from connected components
    :param cells_cc: list of cells from connected components
    :param df_contours: dataframe containing contours
    :return: list of final contours
    """
    # Create dataframes for cells
    df_cells = pl.LazyFrame([{"x1_c": c.x1, "y1_c": c.y1, "x2_c": c.x2, "y2_c": c.y2}
                             for c in cells_cc])

    # Cross join and filters cells contained in contours
    df_contained_cells = (
        df_contours.join(df_cells, how="cross")
        .filter(pl.col("x1_c") >= pl.col("x1"),
                pl.col("y1_c") >= pl.col("y1"),
                pl.col("x2_c") <= pl.col("x2"),
                pl.col("y2_c") <= pl.col("y2"))
        .group_by("id")
        .agg(pl.min("x1_c").alias("x1"),
             pl.min("y1_c").alias("y1"),
             pl.max("x2_c").alias("x2"),
             pl.max("y2_c").alias("y2"))
    )

    # Create final contours
    final_contours = [Cell(x1=row.get('x1'), y1=row.get('y1'), x2=row.get('x2'), y2=row.get('y2'))
                      for row in df_contained_cells.collect().to_dicts()]

    return final_contours


def compute_median_line_sep(img: np.ndarray, cc: np.ndarray,
                            char_length: float) -> Tuple[Optional[float], Optional[List[Cell]]]:
    """
    Compute median separation between rows
    :param img: image array
    :param cc: connected components array
    :param char_length: average character length
    :return: median separation between rows
    """
    # Create image from connected components
    black_img = np.zeros(img.shape, np.uint8)
    cells_cc = [Cell(x1=c[cv2.CC_STAT_LEFT],
                     y1=c[cv2.CC_STAT_TOP],
                     x2=c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH],
                     y2=c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT]) for c in cc]
    for cell in cells_cc:
        cv2.rectangle(black_img, (cell.x1, cell.y1), (cell.x2, cell.y2), (255, 255, 255), -1)

    # Dilate image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(round(char_length)), 1), 1))
    dilate = cv2.dilate(black_img, kernel, iterations=1)

    # Find and map contours
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = list()
    for idx, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        contours.append({"id": idx, "x1": x, "y1": y, "x2": x + w, "y2": y + h})

    if len(contours) == 0:
        return None, []

    # Create contours dataframe
    df_contours = pl.LazyFrame(data=contours)

    # Cross join to get corresponding contours and filter on contours that corresponds horizontally
    df_h_cnts = (df_contours.join(df_contours, how='cross')
                 .filter(pl.col('id') != pl.col('id_right'))
                 .filter(pl.min_horizontal(['x2', 'x2_right']) - pl.max_horizontal(['x1', 'x1_right']) > 0)
                 )

    # Get contour which is directly below
    df_cnts_below = (df_h_cnts.filter(pl.col('y1') < pl.col('y1_right'))
                     .sort(['id', 'y1_right'])
                     .with_columns(pl.lit(1).alias('ones'))
                     .with_columns(pl.col('ones').cum_sum().over(["id"]).alias('rk'))
                     .filter(pl.col('rk') == 1)
                     )

    if df_cnts_below.collect().height == 0:
        return None, [Cell(x1=c.get('x1'), y1=c.get('y1'), x2=c.get('x2'), y2=c.get('y2')) for c in contours]

    # Compute median vertical distance between contours
    median_v_dist = (df_cnts_below.with_columns(((pl.col('y1_right') + pl.col('y2_right')
                                                   - pl.col('y1') - pl.col('y2')) / 2).abs().alias('y_diff'))
                     .select(pl.median('y_diff'))
                     .collect()
                     .to_dicts()
                     .pop()
                     .get('y_diff')
                     )

    # Recompute contours
    final_contours = recompute_contours(cells_cc=cells_cc,
                                        df_contours=df_contours)

    return median_v_dist, final_contours


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

    # Compute median separation between rows
    median_line_sep, contours = compute_median_line_sep(img=img, cc=cc_array, char_length=char_length)

    return char_length, median_line_sep, contours
