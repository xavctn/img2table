# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.column_delimiters.column_groups import \
    vertically_coherent_delimiters, group_delimiters, deduplicate_groups, get_coherent_height, create_delimiter_groups
from img2table.tables.processing.borderless_tables.model import ImageSegment, DelimiterGroup


def test_vertically_coherent_delimiters():
    d_1 = Cell(x1=0, x2=10, y1=0, y2=100)
    d_2 = Cell(x1=0, x2=10, y1=47, y2=128)
    d_3 = Cell(x1=0, x2=10, y1=100, y2=200)

    assert vertically_coherent_delimiters(d_1, d_2)
    assert not vertically_coherent_delimiters(d_1, d_3)
    assert not vertically_coherent_delimiters(d_3, d_2)


def test_group_delimiters():
    delimiters = [Cell(x1=1129, y1=45, x2=1133, y2=1121),
                  Cell(x1=342, y1=45, x2=348, y2=1147),
                  Cell(x1=470, y1=45, x2=565, y2=1147),
                  Cell(x1=651, y1=45, x2=724, y2=1147),
                  Cell(x1=1233, y1=45, x2=1277, y2=1147),
                  Cell(x1=53, y1=45, x2=93, y2=1147),
                  Cell(x1=978, y1=45, x2=999, y2=1121),
                  Cell(x1=226, y1=45, x2=233, y2=1147),
                  Cell(x1=776, y1=45, x2=841, y2=1147),
                  Cell(x1=0, y1=0, x2=20, y2=100),
                  Cell(x1=50, y1=0, x2=100, y2=100),
                  ]
    
    result = group_delimiters(delimiters=delimiters)

    assert len(result) == 1
    assert len(result[0].delimiters) == 9
    assert (result[0].x1, result[0].y1, result[0].x2, result[0].y2) == (53, 45, 1277, 1147)


def test_deduplicate_groups():
    delimiter_groups = [DelimiterGroup(delimiters=[Cell(x1=0, y1=0, x2=10, y2=100),
                                                   Cell(x1=100, y1=0, x2=110, y2=100)]),
                        DelimiterGroup(delimiters=[Cell(x1=40, y1=80, x2=50, y2=120),
                                                   Cell(x1=140, y1=80, x2=150, y2=120)]),
                        DelimiterGroup(delimiters=[Cell(x1=200, y1=0, x2=210, y2=200),
                                                   Cell(x1=300, y1=0, x2=310, y2=200)])
                        ]
    result = deduplicate_groups(delimiter_groups=delimiter_groups)

    assert result == [DelimiterGroup(delimiters=[Cell(x1=200, y1=0, x2=210, y2=200),
                                                 Cell(x1=300, y1=0, x2=310, y2=200)])
                      ]


def test_get_coherent_height():
    with open("test_data/image_segment.json", "r") as f:
        data = json.load(f)
    img_segment = ImageSegment(x1=data.get('x1'),
                               y1=data.get('y1'),
                               x2=data.get('x2'),
                               y2=data.get('y2'),
                               elements=[Cell(**c) for c in data.get('elements')])

    delimiters = [Cell(x1=1129, y1=45, x2=1133, y2=1121),
                  Cell(x1=342, y1=45, x2=348, y2=1192),
                  Cell(x1=470, y1=45, x2=565, y2=1147),
                  Cell(x1=651, y1=45, x2=724, y2=1147),
                  Cell(x1=1233, y1=45, x2=1277, y2=1147),
                  Cell(x1=53, y1=45, x2=93, y2=1147),
                  Cell(x1=978, y1=45, x2=999, y2=1121),
                  Cell(x1=226, y1=45, x2=233, y2=1147),
                  Cell(x1=776, y1=45, x2=841, y2=1147)]

    delimiter_group = DelimiterGroup(delimiters=delimiters)

    result = get_coherent_height(delimiter_group=delimiter_group,
                                 segment=img_segment,
                                 delimiters=delimiters)

    assert (result.x1, result.y1, result.x2, result.y2) == (53, 45, 1277, 1147)
    assert len(result.elements) == 164


def test_create_delimiter_groups():
    with open("test_data/image_segment.json", "r") as f:
        data = json.load(f)
    img_segment = ImageSegment(x1=data.get('x1'),
                               y1=data.get('y1'),
                               x2=data.get('x2'),
                               y2=data.get('y2'),
                               elements=[Cell(**c) for c in data.get('elements')])

    delimiters = [Cell(x1=1129, y1=45, x2=1133, y2=1121),
                  Cell(x1=342, y1=45, x2=348, y2=1192),
                  Cell(x1=470, y1=45, x2=565, y2=1147),
                  Cell(x1=651, y1=45, x2=724, y2=1147),
                  Cell(x1=1233, y1=45, x2=1277, y2=1147),
                  Cell(x1=53, y1=45, x2=93, y2=1147),
                  Cell(x1=978, y1=45, x2=999, y2=1121),
                  Cell(x1=226, y1=45, x2=233, y2=1147),
                  Cell(x1=776, y1=45, x2=841, y2=1147)]

    result = create_delimiter_groups(segment=img_segment,
                                     delimiters=delimiters)

    assert len(result) == 1
    assert len(result[0].delimiters) == 9
    assert (result[0].x1, result[0].y1, result[0].x2, result[0].y2) == (53, 45, 1277, 1147)
    assert len(result[0].elements) == 164
