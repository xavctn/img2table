# coding: utf-8
import json

import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.layout import get_image_elements


def test_get_image_elements():
    thresh = cv2.imread("test_data/test.bmp", cv2.IMREAD_GRAYSCALE)

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = get_image_elements(thresh=thresh,
                                lines=lines,
                                char_length=6.01,
                                median_line_sep=16)

    with open("test_data/elements.json", "r") as f:
        expected = [Cell(**el) for el in json.load(f)]

    assert result == expected
