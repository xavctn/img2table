# coding: utf-8
import json

import cv2

from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables import identify_borderless_tables


def test_identify_borderless_tables():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = identify_borderless_tables(img=img,
                                        char_length=7.24,
                                        median_line_sep=66,
                                        lines=lines,
                                        existing_tables=[])

    assert len(result) == 1
    assert result[0].nb_rows == 16
    assert result[0].nb_columns == 8
    assert (result[0].x1, result[0].y1, result[0].x2, result[0].y2) == (93, 45, 1233, 1060)
