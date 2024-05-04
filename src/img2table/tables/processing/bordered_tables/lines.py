# coding: utf-8
from typing import List, Optional

import cv2
import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line


def identify_straight_lines(canny: np.ndarray, min_line_length: float, char_length: float,
                            vertical: bool = True) -> List[Line]:
    """
    Identify straight lines in image in a specific direction
    :param canny: canny image
    :param min_line_length: minimum line length
    :param char_length: average character length
    :param vertical: boolean indicating if vertical lines are detected
    :return: list of detected lines
    """
    # Apply masking on image
    kernel_dims = (1, round(min_line_length / 2)) if vertical else (round(min_line_length / 2), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
    mask = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply closing for hollow lines
    hollow_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1) if vertical else (1, 3))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, hollow_kernel)

    # Apply closing for dotted lines
    dotted_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, round(min_line_length / 2)) if vertical else (round(min_line_length / 2), 1))
    mask_dotted = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, dotted_kernel)

    # Get stats
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask_dotted, 8, cv2.CV_32S)

    lines = list()
    # Get relevant CC that correspond to lines
    for idx, stat in enumerate(stats):
        if idx == 0:
            continue

        # Get stats
        x, y, w, h, area = stat

        # Filter on aspect ratio
        if max(w, h) / min(w, h) < 5 and min(w, h) >= char_length:
            continue
        # Filter on length
        if max(w, h) < min_line_length:
            continue

        if w >= h:
            line = Line(x1=x, y1=y + h // 2, x2=x + w, y2=y + h // 2, thickness=h)
        else:
            line = Line(x1=x + w // 2, y1=y, x2=x + w // 2, y2=y + h, thickness=w)
        lines.append(line)

    return lines


def detect_lines(img: np.ndarray, contours: Optional[List[Cell]], char_length: Optional[float],
                 min_line_length: Optional[float]) -> (List[Line], List[Line]):
    """
    Detect horizontal and vertical rows on image
    :param img: image array
    :param contours: list of image contours as cell objects
    :param char_length: average character length
    :param min_line_length: minimum line length
    :return: horizontal and vertical rows
    """
    # Create canny image
    canny = cv2.Canny(img, 40, 80, 5)

    # Remove contours from canny image
    for c in contours:
        canny[c.y1:c.y2, c.x1:c.x2] = 0

    # Detect lines
    h_lines = identify_straight_lines(canny=canny,
                                      min_line_length=min_line_length,
                                      char_length=char_length,
                                      vertical=False)
    v_lines = identify_straight_lines(canny=canny,
                                      min_line_length=min_line_length,
                                      char_length=char_length,
                                      vertical=True)

    return h_lines, v_lines
