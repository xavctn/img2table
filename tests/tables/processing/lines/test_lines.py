# coding: utf-8
import json

import cv2

from img2table.tables.objects.line import Line
from img2table.tables.processing.lines import overlapping_filter, detect_lines


def test_overlapping_filter():
    # Create lines
    lines = [Line(x1=10, x2=10, y1=10, y2=100),
             Line(x1=11, x2=11, y1=90, y2=120),
             Line(x1=12, x2=12, y1=210, y2=230),
             Line(x1=12, x2=12, y1=235, y2=255),
             Line(x1=20, x2=20, y1=10, y2=100)]

    result = overlapping_filter(lines=lines, max_gap=10)
    expected = [Line(x1=10, x2=10, y1=10, y2=120),
                Line(x1=12, x2=12, y1=210, y2=255),
                Line(x1=20, x2=20, y1=10, y2=100)]

    assert result == expected


def test_detect_lines():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    h_lines, v_lines = detect_lines(image=img,
                                    rho=0.3,
                                    threshold=10,
                                    minLinLength=10,
                                    maxLineGap=10)

    with open("test_data/expected.json", 'r') as f:
        data = json.load(f)
    h_lines_expected = [Line(**el) for el in data.get('h_lines')]
    v_lines_expected = [Line(**el) for el in data.get('v_lines')]

    assert (h_lines, v_lines) == (h_lines_expected, v_lines_expected)
