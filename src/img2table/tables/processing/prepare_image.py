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

    # Get largest detect contour
    largest_contour = Cell(x1=0, x2=0, y1=0, y2=0)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        contour_cell = Cell(x, y, x + w, y + h)

        if contour_cell.width * contour_cell.height > largest_contour.width * largest_contour.height:
            largest_contour = contour_cell

    # Recreate image from blank image by adding largest contour of the original image
    processed_img = np.zeros(img.shape, dtype=np.uint8)
    processed_img.fill(255)

    # Add contour from original image
    cropped_img = img[largest_contour.y1:largest_contour.y2, largest_contour.x1:largest_contour.x2]
    processed_img[largest_contour.y1:largest_contour.y2, largest_contour.x1:largest_contour.x2] = cropped_img

    return processed_img
