# coding: utf-8
import json

import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.bordered_tables.lines import detect_lines, line_from_cluster, merge_lines


def test_line_from_cluster():
    line_cluster = [Line(x1=0, y1=0, x2=20, y2=0, thickness=1),
                    Line(x1=22, y1=1, x2=48, y2=1, thickness=3)]

    result = line_from_cluster(line_cluster=line_cluster)

    assert result == Line(x1=0, y1=1, x2=48, y2=1, thickness=3)

    line_cluster = [Line(x1=10, y1=7, x2=10, y2=32, thickness=3),
                    Line(x1=12, y1=36, x2=12, y2=112, thickness=3)]

    result = line_from_cluster(line_cluster=line_cluster)

    assert result == Line(x1=11, y1=7, x2=11, y2=112, thickness=5)


def test_merge_lines():
    lines = [Line(x1=0, y1=0, x2=20, y2=0, thickness=1),
             Line(x1=22, y1=1, x2=48, y2=1, thickness=3),
             Line(x1=10, y1=7, x2=10, y2=32, thickness=3),
             Line(x1=12, y1=36, x2=12, y2=112, thickness=3)]

    result = merge_lines(lines=lines, max_gap=5)

    assert result == [Line(x1=0, y1=1, x2=48, y2=1, thickness=3), Line(x1=11, y1=7, x2=11, y2=112, thickness=5)]


def test_detect_lines():
    thresh = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    with open("test_data/contours.json", "r") as f:
        contours = [Cell(**el) for el in json.load(f)]

    h_lines, v_lines = detect_lines(thresh=thresh,
                                    contours=contours,
                                    char_length=8.85,
                                    min_line_length=10)

    with open("test_data/expected.json", 'r') as f:
        data = json.load(f)
    h_lines_expected = [Line(**el) for el in data.get('h_lines')]
    v_lines_expected = [Line(**el) for el in data.get('v_lines')]

    h_lines = sorted(h_lines, key=lambda l: (l.x1, l.y1, l.x2, l.y2))
    v_lines = sorted(v_lines, key=lambda l: (l.x1, l.y1, l.x2, l.y2))
    h_lines_expected = sorted(h_lines_expected, key=lambda l: (l.x1, l.y1, l.x2, l.y2))
    v_lines_expected = sorted(v_lines_expected, key=lambda l: (l.x1, l.y1, l.x2, l.y2))

    assert (h_lines, v_lines) == (h_lines_expected, v_lines_expected)
