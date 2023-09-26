# coding: utf_8
import copy
from typing import List, Union, Optional

import cv2
import numpy as np
import polars as pl

from img2table.tables.objects.cell import Cell


def is_contained_cell(inner_cell: Union[Cell, tuple], outer_cell: Union[Cell, tuple], percentage: float = 0.9) -> bool:
    """
    Assert if the inner cell is contained in outer cell
    :param inner_cell: inner cell
    :param outer_cell: Table object
    :param percentage: percentage of the inner cell that needs to be contained in the outer cell
    :return: boolean indicating if the inner cell is contained in the outer cell
    """
    # If needed, convert inner cell to Cell object
    if isinstance(inner_cell, tuple):
        inner_cell = Cell(*inner_cell)
    # If needed, convert outer cell to Cell object
    if isinstance(outer_cell, tuple):
        outer_cell = Cell(*outer_cell)

    # Compute common coordinates
    x_left = max(inner_cell.x1, outer_cell.x1)
    y_top = max(inner_cell.y1, outer_cell.y1)
    x_right = min(inner_cell.x2, outer_cell.x2)
    y_bottom = min(inner_cell.y2, outer_cell.y2)

    # Compute intersection area as well as inner cell area
    intersection_area = max(0, (x_right - x_left)) * max(0, (y_bottom - y_top))

    return intersection_area / inner_cell.area >= percentage


def merge_overlapping_contours(contours: List[Cell]) -> List[Cell]:
    """
    Merge overlapping contours
    :param contours: list of contours as Cell objects
    :return: list of merged contours
    """
    if len(contours) == 0:
        return []

    # Create dataframe with contours
    df_cnt = pl.LazyFrame(data=[{"id": idx, "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2, "area": c.area}
                                for idx, c in enumerate(contours)])

    # Cross join
    df_cross = (df_cnt.join(df_cnt, how='cross')
                .filter(pl.col('id') != pl.col('id_right'))
                .filter(pl.col('area') <= pl.col('area_right'))
                )

    # Compute intersection area between contours and identify if the smallest contour overlaps the largest one
    x_left = pl.max_horizontal('x1', 'x1_right')
    x_right = pl.min_horizontal('x2', 'x2_right')
    y_top = pl.max_horizontal('y1', 'y1_right')
    y_bottom = pl.min_horizontal('y2', 'y2_right')
    intersection = pl.max_horizontal(x_right - x_left, 0) * pl.max_horizontal(y_bottom - y_top, 0)

    df_cross = (df_cross.with_columns(intersection.alias('intersection'))
                .with_columns((pl.col('intersection') / pl.col('area') >= 0.25).alias('overlaps'))
                )

    # Identify relevant contours: no contours is overlapping it
    deleted_contours = df_cross.filter(pl.col('overlaps')).select('id').unique()
    df_overlap = (df_cross.filter(pl.col('overlaps'))
                  .group_by(pl.col('id_right').alias('id'))
                  .agg(pl.min('x1').alias('x1_overlap'),
                       pl.max('x2').alias('x2_overlap'),
                       pl.min('y1').alias('y1_overlap'),
                       pl.max('y2').alias('y2_overlap'))
                  )

    df_final = (df_cnt.join(deleted_contours, on="id", how="anti")
                .join(df_overlap, on='id', how='left')
                .select([pl.min_horizontal('x1', 'x1_overlap').alias('x1'),
                         pl.max_horizontal('x2', 'x2_overlap').alias('x2'),
                         pl.min_horizontal('y1', 'y1_overlap').alias('y1'),
                         pl.max_horizontal('y2', 'y2_overlap').alias('y2'),
                         ])
                )

    # Map results to cells
    return [Cell(**d) for d in df_final.collect().to_dicts()]


def merge_contours(contours: List[Cell], vertically: Optional[bool] = True) -> List[Cell]:
    """
    Create merge contours by an axis
    :param contours: list of contours
    :param vertically: boolean indicating if contours are merged according to the vertical or horizontal axis
    :return: merged contours
    """
    # If contours is empty, return empty list
    if len(contours) == 0:
        return contours

    # If vertically is None, merge only contained contours
    if vertically is None:
        return merge_overlapping_contours(contours=contours)

    # Define dimensions used to merge contours
    idx_1 = "y1" if vertically else "x1"
    idx_2 = "y2" if vertically else "x2"
    sort_idx_1 = "x1" if vertically else "y1"
    sort_idx_2 = "x2" if vertically else "y2"

    # Sort contours
    sorted_cnts = sorted(contours,
                         key=lambda cnt: (getattr(cnt, idx_1), getattr(cnt, idx_2), getattr(cnt, sort_idx_1)))

    # Loop over contours and merge overlapping contours
    seq = iter(sorted_cnts)
    list_cnts = [copy.deepcopy(next(seq))]
    for cnt in seq:
        # If contours overlap, update current contour
        if getattr(cnt, idx_1) <= getattr(list_cnts[-1], idx_2):
            # Update current contour coordinates
            setattr(list_cnts[-1], idx_2, max(getattr(list_cnts[-1], idx_2), getattr(cnt, idx_2)))
            setattr(list_cnts[-1], sort_idx_1, min(getattr(list_cnts[-1], sort_idx_1), getattr(cnt, sort_idx_1)))
            setattr(list_cnts[-1], sort_idx_2, max(getattr(list_cnts[-1], sort_idx_2), getattr(cnt, sort_idx_2)))
        else:
            list_cnts.append(copy.deepcopy(cnt))

    return list_cnts


def get_contours_cell(img: np.ndarray, cell: Cell, margin: int = 5, blur_size: int = 9, kernel_size: int = 15,
                      merge_vertically: Optional[bool] = True) -> List[Cell]:
    """
    Get list of contours contained in cell
    :param img: image array
    :param cell: Cell object
    :param margin: margin in pixels used for cropped images
    :param blur_size: kernel size for blurring operation
    :param kernel_size: kernel size for dilate operation
    :param merge_vertically: boolean indicating if contours are merged according to the vertical or horizontal axis
    :return: list of contours contained in cell
    """
    height, width = img.shape[:2]
    # Get cropped image
    cropped_img = img[max(cell.y1 - margin, 0):min(cell.y2 + margin, height),
                      max(cell.x1 - margin, 0):min(cell.x2 + margin, width)]

    # If cropped image is empty, do not do anything
    height_cropped, width_cropped = cropped_img.shape[:2]
    if height_cropped <= 0 or width_cropped <= 0:
        return []

    # Reprocess images
    blur = cv2.GaussianBlur(cropped_img, (blur_size, blur_size), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Get list of contours
    list_cnts_cell = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x = x + cell.x1 - margin
        y = y + cell.y1 - margin
        contour_cell = Cell(x, y, x + w, y + h)
        list_cnts_cell.append(contour_cell)

    # Add contours to row
    contours = merge_contours(contours=list_cnts_cell,
                              vertically=merge_vertically)

    return contours
