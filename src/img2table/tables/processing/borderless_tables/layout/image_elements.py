# coding: utf-8
from typing import List

import cv2
import numpy as np

from img2table.tables.objects.cell import Cell


def get_image_elements(thresh: np.ndarray, char_length: float, median_line_sep: float) -> List[Cell]:
    """
    Identify image elements
    :param thresh: thresholded image array
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: list of image elements
    """
    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (max(int(char_length), 1), max(int(median_line_sep // 6), 1)))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Get list of contours
    elements = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        elements.append(Cell(x1=x, y1=y, x2=x + w, y2=y + h))

    # Filter elements that are too small
    elements = [el for el in elements if min(el.height, el.width) >= char_length]

    return elements
