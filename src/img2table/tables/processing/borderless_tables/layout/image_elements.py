# coding: utf-8
from typing import List

import cv2
import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.common import merge_contours


def get_image_elements(img: np.ndarray, lines: List[Line], char_length: float,
                       median_line_sep: float, blur_size: int = 3) -> List[Cell]:
    """
    Identify image elements
    :param img: image array
    :param lines: list of image rows
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param blur_size: kernel size for blurring operation
    :return: list of image elements
    """
    # Reprocess image
    blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    thresh = cv2.Canny(blur, 85, 255)

    # Mask rows
    for l in lines:
        if l.horizontal and l.length >= 20 * char_length:
            cv2.rectangle(thresh, (l.x1 - l.thickness, l.y1), (l.x2 + l.thickness, l.y2), (0, 0, 0), 3 * l.thickness)
        elif l.vertical and l.length >= 5 * char_length:
            cv2.rectangle(thresh, (l.x1, l.y1 - l.thickness), (l.x2, l.y2 + l.thickness), (0, 0, 0), 2 * l.thickness)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (max(int(1.5 * char_length), 1), max(int(median_line_sep // 6), 1)))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Get list of contours
    elements = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        elements.append(Cell(x1=x, y1=y, x2=x + w, y2=y + h))

    # Merge elements
    elements = merge_contours(contours=elements,
                              vertically=None)

    return elements
