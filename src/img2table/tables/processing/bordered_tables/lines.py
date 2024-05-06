# coding: utf-8
from typing import List, Optional

import cv2
import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line


def identify_straight_lines(thresh: np.ndarray, min_line_length: float, char_length: float,
                            vertical: bool = True) -> List[Line]:
    """
    Identify straight lines in image in a specific direction
    :param thresh: thresholded edge image
    :param min_line_length: minimum line length
    :param char_length: average character length
    :param vertical: boolean indicating if vertical lines are detected
    :return: list of detected lines
    """
    # Apply masking on image
    kernel_dims = (1, round(min_line_length / 3)) if vertical else (round(min_line_length / 3), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply closing for hollow lines
    hollow_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1) if vertical else (1, 3))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, hollow_kernel)

    # Apply closing for dotted lines
    dotted_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, round(min_line_length / 6)) if vertical else (round(min_line_length / 6), 1))
    mask_dotted = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, dotted_kernel)

    # Apply masking on line length
    kernel_dims = (1, min_line_length) if vertical else (min_line_length, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
    final_mask = cv2.morphologyEx(mask_dotted, cv2.MORPH_OPEN, kernel, iterations=1)

    # Get stats
    _, _, stats, _ = cv2.connectedComponentsWithStats(final_mask, 8, cv2.CV_32S)

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
            non_blank_pixels = np.where(np.sum(thresh[y:y+h, x:x+w], axis=0) > 0)
            line = Line(x1=x + np.min(non_blank_pixels),
                        y1=y + h // 2,
                        x2=x + np.max(non_blank_pixels),
                        y2=y + h // 2,
                        thickness=h)
        else:
            non_blank_pixels = np.where(np.sum(thresh[y:y+h, x:x+w], axis=1) > 0)
            line = Line(x1=x + w // 2,
                        y1=y + np.min(non_blank_pixels),
                        x2=x + w // 2,
                        y2=y + np.max(non_blank_pixels),
                        thickness=w)
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
    # Grayscale and blurring
    blur = cv2.bilateralFilter(img, 3, 40, 80)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

    # Apply laplacian and filter image
    laplacian = cv2.Laplacian(src=gray, ksize=3, ddepth=cv2.CV_64F)
    edge_img = cv2.convertScaleAbs(laplacian)

    # Remove contours and convert to binary image
    for c in contours:
        edge_img[c.y1 - 1:c.y2 + 1, c.x1 - 1:c.x2 + 1] = 0
    binary_img = 255 * (edge_img >= min(3 * np.mean(edge_img), np.max(edge_img))).astype(np.uint8)

    # Detect lines
    h_lines = identify_straight_lines(thresh=binary_img,
                                      min_line_length=min_line_length,
                                      char_length=char_length,
                                      vertical=False)
    v_lines = identify_straight_lines(thresh=binary_img,
                                      min_line_length=min_line_length,
                                      char_length=char_length,
                                      vertical=True)

    return h_lines, v_lines
