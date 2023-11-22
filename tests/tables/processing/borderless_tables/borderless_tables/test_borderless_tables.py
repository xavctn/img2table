# coding: utf-8
import json

import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables import identify_borderless_tables


def test_identify_borderless_tables():
    thresh = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    with open("test_data/contours.json", 'r') as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = identify_borderless_tables(thresh=thresh,
                                        char_length=7.13,
                                        median_line_sep=66,
                                        lines=lines,
                                        contours=contours,
                                        existing_tables=[])

    assert len(result) == 1
    assert result[0].nb_rows == 16
    assert result[0].nb_columns == 9
    assert (result[0].x1, result[0].y1, result[0].x2, result[0].y2) == (134, 45, 1155, 1060)
