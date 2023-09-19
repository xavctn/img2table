# coding: utf-8
import json

import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables import segment_image


def test_segment_image():
    img = cv2.imread("test_data/test.png", 0)

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    with open("test_data/contours.json", 'r') as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = segment_image(img=img,
                           lines=lines,
                           char_length=14,
                           median_line_sep=85.75,
                           contours=contours)

    assert len(result) == 1
    assert len(result[0].elements) == 20
    assert len(result[0].table_areas) == 2
    assert len(result[0].whitespaces) == 8
