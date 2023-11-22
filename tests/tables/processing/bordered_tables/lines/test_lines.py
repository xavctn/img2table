# coding: utf-8
import json

import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.bordered_tables.lines import overlapping_filter, detect_lines, remove_word_lines, \
    create_lines_from_intersection


def test_overlapping_filter():
    # Create rows
    lines = [Line(x1=10, x2=10, y1=10, y2=100),
             Line(x1=11, x2=11, y1=90, y2=120),
             Line(x1=12, x2=12, y1=210, y2=230),
             Line(x1=12, x2=12, y1=235, y2=255),
             Line(x1=20, x2=20, y1=10, y2=100)]

    result = overlapping_filter(lines=lines, max_gap=10)
    expected = [Line(x1=10, x2=10, y1=10, y2=120, thickness=1),
                Line(x1=12, x2=12, y1=210, y2=255, thickness=1),
                Line(x1=20, x2=20, y1=10, y2=100, thickness=1)]

    assert result == expected


def test_create_lines_from_intersection():
    line_dict = {"x1_line": 100,
                 "x2_line": 100,
                 "y1_line": 10,
                 "y2_line": 30,
                 "vertical": True,
                 "thickness": 2,
                 "intersecting": None}
    assert create_lines_from_intersection(line_dict=line_dict) == [Line(x1=100, x2=100, y1=10, y2=30, thickness=2)]

    line_dict = {"x1_line": 100,
                 "x2_line": 100,
                 "y1_line": 10,
                 "y2_line": 30,
                 "vertical": True,
                 "thickness": 2,
                 "intersecting": [{"x1": 80, "x2": 120, "y1": 0, "y2": 15},
                                  {"x1": 80, "x2": 120, "y1": 21, "y2": 28}]
                 }
    assert create_lines_from_intersection(line_dict=line_dict) == [Line(x1=100, x2=100, y1=16, y2=20, thickness=2),
                                                                   Line(x1=100, x2=100, y1=29, y2=30, thickness=2)]

    line_dict = {"x1_line": 230,
                 "x2_line": 273,
                 "y1_line": 10,
                 "y2_line": 10,
                 "vertical": False,
                 "thickness": 3,
                 "intersecting": [{"x1": 200, "x2": 250, "y1": 0, "y2": 15},
                                  {"x1": 250, "x2": 300, "y1": 1, "y2": 28}]
                 }
    assert create_lines_from_intersection(line_dict=line_dict) == []


def test_remove_word_lines():
    with open("test_data/contours.json", "r") as f:
        contours = [Cell(**el) for el in json.load(f)]
    lines = [Line(x1=10, x2=10, y1=10, y2=100, thickness=1),
             Line(x1=975, x2=975, y1=40, y2=60, thickness=1)]

    result = remove_word_lines(lines=lines, contours=contours)

    result = sorted(result, key=lambda l: (l.x1, l.y1, l.x2, l.y2))
    assert result == [Line(x1=10, x2=10, y1=10, y2=100, thickness=1),
                      Line(x1=975, y1=40, x2=975, y2=40, thickness=1),
                      Line(x1=975, y1=56, x2=975, y2=60, thickness=1)]


def test_detect_lines():
    thresh = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    with open("test_data/contours.json", "r") as f:
        contours = [Cell(**el) for el in json.load(f)]

    h_lines, v_lines = detect_lines(thresh=thresh,
                                    rho=0.3,
                                    threshold=10,
                                    minLinLength=10,
                                    maxLineGap=10,
                                    contours=contours,
                                    char_length=8.85)

    with open("test_data/expected.json", 'r') as f:
        data = json.load(f)
    h_lines_expected = [Line(**el) for el in data.get('h_lines')]
    v_lines_expected = [Line(**el) for el in data.get('v_lines')]

    h_lines = sorted(h_lines, key=lambda l: (l.x1, l.y1, l.x2, l.y2))
    v_lines = sorted(v_lines, key=lambda l: (l.x1, l.y1, l.x2, l.y2))
    h_lines_expected = sorted(h_lines_expected, key=lambda l: (l.x1, l.y1, l.x2, l.y2))
    v_lines_expected = sorted(v_lines_expected, key=lambda l: (l.x1, l.y1, l.x2, l.y2))

    assert (h_lines, v_lines) == (h_lines_expected, v_lines_expected)
