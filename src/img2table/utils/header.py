# coding: utf-8

import numpy as np
import copy

from img2table.objects.ocr import OCRPage


def brightness_img(img) -> float:
    """
    Compute brightness value of image
    :param img: image array
    :return: brightness value of the image
    """
    return np.average(np.linalg.norm(img, axis=2)) / np.sqrt(3)


def detect_header(img: np.ndarray, table, ocr_page: OCRPage) -> bool:
    """
    Detect if a table has a header
    :param img: image array
    :param table: Table object
    :param ocr_page: OCRPage object
    :return: boolean indicating if the table has an header
    """
    from img2table.objects.tables import Table

    # If table has only one row, no headers
    if table.nb_rows <= 1:
        return False

    # Create first row and "body" tables
    header_table = Table(rows=table.items[0])
    body_table = Table(rows=table.items[1:])

    # Compute brightness of both table
    cropped_img_header = copy.deepcopy(img)[header_table.y1:header_table.y2, header_table.x1:header_table.x2]
    brightness_header = brightness_img(img=cropped_img_header)

    cropped_img_body = copy.deepcopy(img)[body_table.y1:body_table.y2, body_table.x1:body_table.x2]
    brightness_body = brightness_img(img=cropped_img_body)

    # If brightness difference is large, the first row is a header
    if abs(brightness_header / brightness_body - 1) >= 0.2:
        return True

    # Compute average text size
    txt_size_header = header_table.get_text_size(ocr_page=ocr_page)
    txt_size_body = body_table.get_text_size(ocr_page=ocr_page)

    # If text size difference is large, the first row is a header
    if txt_size_header is not None and txt_size_body is not None:
        if abs(txt_size_header / txt_size_body - 1) >= 0.2:
            return True

    return False
