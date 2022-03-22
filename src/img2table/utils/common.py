# coding: utf_8
from typing import List

import numpy as np
from cv2 import cv2

from img2table.objects.tables import Cell, Table


def merge_contours(contours: List[Cell], vertically: bool = True) -> List[Cell]:
    """
    Create merge contours by an axis
    :param contours: list of contours
    :param vertically: boolean indicating if contours are merged according to the vertical or horizontal axis
    :return: merged contours
    """
    # If contours is empty, return empty list
    if len(contours) == 0:
        return contours

    # Define dimensions used to merge contours
    if vertically:
        idx_1 = "y1"
        idx_2 = "y2"
        sort_idx_1 = "x1"
        sort_idx_2 = "x2"
    else:
        idx_1 = "x1"
        idx_2 = "x2"
        sort_idx_1 = "y1"
        sort_idx_2 = "y2"

    # Sort contours
    sorted_cnts = sorted(contours,
                         key=lambda cnt: (getattr(cnt, idx_1), getattr(cnt, idx_2), getattr(cnt, sort_idx_1)))

    list_cnts = list()
    # Loop over contours and merge overlapping contours
    for idx, cnt in enumerate(sorted_cnts):
        if idx == 0:
            curr_cnt = cnt
        else:
            # If contours overlap, update current contour
            if getattr(cnt, idx_1) <= getattr(curr_cnt, idx_2):
                # Update current contour coordinates
                setattr(curr_cnt, idx_2, max(getattr(curr_cnt, idx_2), getattr(cnt, idx_2)))
                setattr(curr_cnt, sort_idx_1, min(getattr(curr_cnt, sort_idx_1), getattr(cnt, sort_idx_1)))
                setattr(curr_cnt, sort_idx_2, max(getattr(curr_cnt, sort_idx_2), getattr(cnt, sort_idx_2)))
            # Else, add current contour and open a new one
            else:
                list_cnts.append(curr_cnt)
                curr_cnt = cnt

    list_cnts.append(curr_cnt)

    return list_cnts


def get_bounding_area_text(img: np.ndarray, table: Table, margin: int = 5, blur_size: int = 9,
                           kernel_size: int = 15, merge_vertically: bool = True) -> Table:
    """
    Compute list of text bounding areas for each row of a table
    :param img: image array
    :param table: Table object
    :param margin: margin in pixels used for cropped images
    :param blur_size: kernel size for blurring operation
    :param kernel_size: kernel size for dilate operation
    :param merge_vertically: boolean indicating if contours are merged according to the vertical or horizontal axis
    :return: Table object with contours associated with each row
    """
    height, width, _ = img.shape

    # Iter over rows
    for idx, row in enumerate(table.items):
        cropped_img = img[max(row.y1 - margin, 0):min(row.y2 + margin, height),
                      max(row.x1 - margin, 0):min(row.x2 + margin, width)]

        # If cropped image is empty, do not do anything
        height_cropped, width_cropped, _ = cropped_img.shape
        if height_cropped <= 0 or width_cropped <= 0:
            continue

        # Reprocess images
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

        # Dilate to combine adjacent text contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Get list of contours
        list_cnts_row = list()
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            x = x + row.x1 - margin
            y = y + row.y1 - margin
            contour_cell = Cell(x, y, x + w, y + h)
            list_cnts_row.append(contour_cell)

        # Add contours to row
        table.items[idx].add_contours(merge_contours(contours=list_cnts_row,
                                                     vertically=merge_vertically),
                                      replace=True)

    return table
