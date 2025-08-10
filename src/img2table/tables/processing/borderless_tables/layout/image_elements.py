
import cv2
import numpy as np

from img2table.tables.objects.cell import Cell


def get_image_elements(thresh: np.ndarray, char_length: float) -> list[Cell]:
    """
    Identify image elements
    :param thresh: thresholded image array
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: list of image elements
    """
    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Get list of contours
    elements = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if ((min(h, w) >= 0.5 * char_length and max(h, w) >= char_length)
                or (w / h >= 2 and 0.5 * char_length <= w <= 1.5 * char_length)):
            elements.append(Cell(x1=x, y1=y, x2=x + w, y2=y + h))

    return elements
