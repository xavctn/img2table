# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.lines import identify_lines, create_h_pos_groups, \
    vertically_coherent_groups, is_text_block, identify_line_groups
from img2table.tables.processing.borderless_tables.model import ImageSegment, TableLine, LineGroup


def test_identify_lines():
    with open("test_data/image_segment.json", "r") as f:
        data = json.load(f)
    img_segment = ImageSegment(x1=data.get('x1'),
                               y1=data.get('y1'),
                               x2=data.get('x2'),
                               y2=data.get('y2'),
                               elements=[Cell(**c) for c in data.get('elements')])

    result = identify_lines(elements=img_segment.elements,
                            ref_size=12)

    assert len(result) == 6
    assert all([isinstance(el, TableLine) for el in result])


def test_create_h_pos_groups():
    lines = [TableLine(cells=[Cell(x1=0, x2=20, y1=0, y2=10)]),
             TableLine(cells=[Cell(x1=0, x2=20, y1=20, y2=30)]),
             TableLine(cells=[Cell(x1=19, x2=40, y1=3, y2=12)]),
             TableLine(cells=[Cell(x1=100, x2=120, y1=0, y2=10)]),
             TableLine(cells=[Cell(x1=100, x2=120, y1=20, y2=30)])]

    result = create_h_pos_groups(lines=lines)

    expected = [
        [TableLine(cells=[Cell(x1=0, x2=20, y1=0, y2=10)]),
         TableLine(cells=[Cell(x1=0, x2=20, y1=20, y2=30)]),
         TableLine(cells=[Cell(x1=19, x2=40, y1=3, y2=12)])],
        [TableLine(cells=[Cell(x1=100, x2=120, y1=0, y2=10)]),
         TableLine(cells=[Cell(x1=100, x2=120, y1=20, y2=30)])]
    ]

    assert result == expected


def test_vertically_coherent_groups():
    lines = [TableLine(cells=[Cell(x1=0, x2=20, y1=0, y2=10), Cell(x1=20, x2=40, y1=0, y2=10)]),
             TableLine(cells=[Cell(x1=0, x2=20, y1=13, y2=23), Cell(x1=20, x2=40, y1=13, y2=23)]),
             TableLine(cells=[Cell(x1=0, x2=20, y1=30, y2=40), Cell(x1=20, x2=40, y1=30, y2=40)]),
             TableLine(cells=[Cell(x1=0, x2=20, y1=50, y2=60), Cell(x1=20, x2=40, y1=50, y2=60)]),
             TableLine(cells=[Cell(x1=0, x2=20, y1=63, y2=73), Cell(x1=20, x2=40, y1=63, y2=73)])]

    result = vertically_coherent_groups(lines=lines, max_gap=5)

    expected = [
        LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=20, y1=0, y2=10), Cell(x1=20, x2=40, y1=0, y2=10)]),
                         TableLine(cells=[Cell(x1=0, x2=20, y1=13, y2=23), Cell(x1=20, x2=40, y1=13, y2=23)])]),
        LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=20, y1=50, y2=60), Cell(x1=20, x2=40, y1=50, y2=60)]),
                         TableLine(cells=[Cell(x1=0, x2=20, y1=63, y2=73), Cell(x1=20, x2=40, y1=63, y2=73)])])
    ]

    assert result == expected


def test_is_text_block():
    line_group = LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=20, y1=0, y2=10), Cell(x1=20, x2=40, y1=0, y2=10)]),
                                  TableLine(cells=[Cell(x1=0, x2=20, y1=13, y2=23), Cell(x1=20, x2=40, y1=13, y2=23)])])
    assert is_text_block(line_group=line_group, char_length=8.44)

    line_group = LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=20, y1=0, y2=10), Cell(x1=40, x2=60, y1=0, y2=10)]),
                                  TableLine(cells=[Cell(x1=0, x2=20, y1=13, y2=23), Cell(x1=40, x2=60, y1=13, y2=23)])])
    assert not is_text_block(line_group=line_group, char_length=8.44)


def test_identify_line_groups():
    with open("test_data/image_segment.json", "r") as f:
        data = json.load(f)
    img_segment = ImageSegment(x1=data.get('x1'),
                               y1=data.get('y1'),
                               x2=data.get('x2'),
                               y2=data.get('y2'),
                               elements=[Cell(**c) for c in data.get('elements')])

    result = identify_line_groups(segment=img_segment,
                                  median_line_sep=51,
                                  char_length=8.44)

    assert isinstance(result, ImageSegment)
    assert result.x1 == img_segment.x1
    assert result.y1 == img_segment.y1
    assert result.x2 == img_segment.x2
    assert result.y2 == img_segment.y2
    assert result.elements == img_segment.elements
    assert len(result.line_groups) == 1
    assert len(result.line_groups[0].lines) == 6
