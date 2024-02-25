# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import ImageSegment
from img2table.tables.processing.borderless_tables.whitespaces import get_whitespaces, adjacent_whitespaces, \
    identify_coherent_v_whitespaces, get_relevant_vertical_whitespaces


def test_get_whitespaces():
    with open("test_data/image_segment.json", "r") as f:
        data = json.load(f)
    img_segment = ImageSegment(x1=data.get('x1'),
                               y1=data.get('y1'),
                               x2=data.get('x2'),
                               y2=data.get('y2'),
                               elements=[Cell(**c) for c in data.get('elements')])

    result = get_whitespaces(segment=img_segment, vertical=True)

    assert len(result) == 38


def test_adjacent_whitespaces():
    c_1 = Cell(x1=0, x2=10, y1=0, y2=10)
    c_2 = Cell(x1=10, x2=20, y1=0, y2=10)
    c_3 = Cell(x1=10, x2=20, y1=0, y2=20)
    c_4 = Cell(x1=20, x2=30, y1=0, y2=10)

    assert adjacent_whitespaces(c_1, c_2)
    assert adjacent_whitespaces(c_1, c_3)
    assert not adjacent_whitespaces(c_1, c_4)


def test_identify_coherent_v_whitespaces():
    v_whitespaces = [Cell(x1=0, x2=10, y1=0, y2=10),
                     Cell(x1=10, x2=20, y1=0, y2=20),
                     Cell(x1=20, x2=30, y1=0, y2=10),
                     Cell(x1=50, x2=60, y1=0, y2=20),
                     Cell(x1=60, x2=70, y1=0, y2=18),
                     Cell(x1=70, x2=80, y1=0, y2=10),
                     Cell(x1=80, x2=90, y1=0, y2=20),
                     Cell(x1=100, x2=110, y1=0, y2=10)]

    result = identify_coherent_v_whitespaces(v_whitespaces=v_whitespaces)

    expected = [Cell(x1=10, x2=20, y1=0, y2=20),
                Cell(x1=50, x2=60, y1=0, y2=20),
                Cell(x1=80, x2=90, y1=0, y2=20),
                Cell(x1=100, x2=110, y1=0, y2=10)]

    assert set(result) == set(expected)


def test_get_relevant_vertical_whitespaces():
    with open("test_data/image_segment.json", "r") as f:
        data = json.load(f)
    img_segment = ImageSegment(x1=data.get('x1'),
                               y1=data.get('y1'),
                               x2=data.get('x2'),
                               y2=data.get('y2'),
                               elements=[Cell(**c) for c in data.get('elements')])

    result = get_relevant_vertical_whitespaces(segment=img_segment,
                                               char_length=7.0,
                                               median_line_sep=14)

    assert len(result) == 11
