# coding: utf_8
import copy
from typing import List, Union, Optional

import cv2
import numpy as np

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

    if x_right < x_left or y_bottom < y_top:
        return False

    # Compute intersection area as well as inner cell area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    inner_cell_area = inner_cell.height * inner_cell.width

    return intersection_area / inner_cell_area >= percentage


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
        sorted_cnt = sorted(contours, key=lambda cnt: cnt.height * cnt.width, reverse=True)

        seq = iter(sorted_cnt)
        list_cnts = [copy.deepcopy(next(seq))]
        for cnt in seq:
            contained_cnt = [idx for idx, el in enumerate(list_cnts)
                             if is_contained_cell(inner_cell=cnt, outer_cell=el, percentage=0.75)]
            if len(contained_cnt) == 1:
                id = contained_cnt.pop()
                list_cnts[id].x1 = min(list_cnts[id].x1, cnt.x1)
                list_cnts[id].y1 = min(list_cnts[id].y1, cnt.y1)
                list_cnts[id].x2 = max(list_cnts[id].x2, cnt.x2)
                list_cnts[id].y2 = max(list_cnts[id].y2, cnt.y2)
            else:
                list_cnts.append(copy.deepcopy(cnt))

        return list_cnts

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
                      merge_vertically: bool = True) -> List[Cell]:
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
