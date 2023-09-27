# coding: utf-8
import json

import cv2

from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables import segment_image


def test_segment_image():
    img = cv2.imread("test_data/test.bmp", 0)

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = segment_image(img=img,
                           lines=lines,
                           char_length=5.04,
                           median_line_sep=16)

    assert len(result) == 3

    assert len(result[0].elements) == 44
    assert len(result[0].table_areas) == 1
    assert len(result[0].whitespaces) == 7

    assert len(result[1].elements) == 37
    assert len(result[1].table_areas) == 1
    assert len(result[1].whitespaces) == 6

    assert len(result[2].elements) == 40
    assert len(result[2].table_areas) == 1
    assert len(result[2].whitespaces) == 7
