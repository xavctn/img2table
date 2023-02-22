# coding: utf-8

import cv2
import numpy as np

from img2table.tables.objects.cell import Cell


def prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Prepare image by removing background and keeping a white base
    :param img: original image array
    :return: processed image
    """
    # Preprocess image
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dilation = cv2.dilate(thresh, (10, 10), iterations=3)

    # Compute contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get contours cells
    contour_cells = list()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        contour_cells.append(Cell(x, y, x + w, y + h))
    contour_cells = sorted(contour_cells, key=lambda c: c.area, reverse=True)

    if contour_cells:
        largest_contour = None
        if len(contour_cells) == 1:
            # Set largest contour
            largest_contour = contour_cells.pop(0)
        elif contour_cells[0].area / contour_cells[1].area > 10:
            # Set largest contour
            largest_contour = contour_cells.pop(0)

        if largest_contour:
            # Recreate image from blank image by adding largest contour of the original image
            processed_img = np.zeros(img.shape, dtype=np.uint8)
            processed_img.fill(255)

            # Add contour from original image
            cropped_img = img[largest_contour.y1:largest_contour.y2, largest_contour.x1:largest_contour.x2]
            processed_img[largest_contour.y1:largest_contour.y2, largest_contour.x1:largest_contour.x2] = cropped_img

            return processed_img

    return img
