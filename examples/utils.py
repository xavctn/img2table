# coding: utf-8

import cv2
import numpy as np

from img2table.document import Image
from img2table.ocr.base import OCRInstance


def display_borderless_tables(img: Image, ocr: OCRInstance) -> np.ndarray:
    """
    Create display of borderless table extraction
    :param img: Image object
    :param ocr: OCRInstance object
    :return: display image
    """
    # Extract tables
    extracted_tables = img.extract_tables(ocr=ocr,
                                          borderless_tables=True)

    # Create image displaying extracted tables
    display_image = cv2.cvtColor(list(img.images)[0], cv2.COLOR_GRAY2RGB)
    for tb in extracted_tables:
        for row in tb.content.values():
            for cell in row:
                cv2.rectangle(display_image, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2),
                              (255, 0, 0), 2)

    # Create white separator image
    width = min(display_image.shape[1] // 10, 100)
    white_img = cv2.cvtColor(255 * np.ones((display_image.shape[0], width), dtype=np.uint8), cv2.COLOR_GRAY2RGB)

    # Stack images
    final_image = np.hstack([cv2.cvtColor(list(img.images)[0], cv2.COLOR_GRAY2RGB),
                             white_img,
                             display_image])

    return final_image
