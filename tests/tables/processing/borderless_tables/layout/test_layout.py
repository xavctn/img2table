# coding: utf-8
import json

import cv2

from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables import segment_image


def test_segment_image():
    thresh = cv2.imread("test_data/test.bmp", 0)

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = segment_image(thresh=thresh,
                           lines=lines,
                           char_length=6.01,
                           median_line_sep=16)

    assert len(result) == 4

    assert len(result[0].elements) == 26
    assert len(result[0].table_areas) == 1
    assert len(result[0].whitespaces) == 5

    assert len(result[1].elements) == 46
    assert len(result[1].table_areas) == 1
    assert len(result[1].whitespaces) == 6

    assert len(result[2].elements) == 35
    assert len(result[2].table_areas) == 1
    assert len(result[2].whitespaces) == 7

    assert len(result[3].elements) == 32
    assert len(result[3].table_areas) == 1
    assert len(result[3].whitespaces) == 6
