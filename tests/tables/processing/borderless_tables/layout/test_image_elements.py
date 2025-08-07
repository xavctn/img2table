# coding: utf-8
import json

import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.layout import get_image_elements


def test_get_image_elements():
    thresh = cv2.imread("test_data/text_thresh.bmp", cv2.IMREAD_GRAYSCALE)

    result = get_image_elements(thresh=thresh,
                                char_length=6.0)

    with open("test_data/elements.json", "r") as f:
        expected = [Cell(**el) for el in json.load(f)]

    assert result == expected
