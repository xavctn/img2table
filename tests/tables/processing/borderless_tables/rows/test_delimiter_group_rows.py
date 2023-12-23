# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup
from img2table.tables.processing.borderless_tables.rows.delimiter_group_rows import \
    identify_delimiter_group_rows, identify_row_delimiters, filter_coherent_row_delimiters, correct_delimiter_width


def test_identify_row_delimiters():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    result = identify_row_delimiters(delimiter_group=delimiter_group)

    with open("test_data/h_whitespaces.json", "r") as f:
        expected = [Cell(**c) for c in json.load(f)]

    assert result == expected


def test_filter_coherent_row_delimiters():
    row_delimiters = [Cell(x1=0, x2=100, y1=0, y2=0),
                      Cell(x1=0, x2=80, y1=10, y2=10),
                      Cell(x1=0, x2=100, y1=20, y2=20)]

    delimiter_group = DelimiterGroup(delimiters=[Cell(x1=0, x2=0, y1=0, y2=20),
                                                 Cell(x1=30, x2=30, y1=0, y2=20),
                                                 Cell(x1=60, x2=60, y1=0, y2=20),
                                                 Cell(x1=100, x2=100, y1=0, y2=20)],
                                     elements=[Cell(x1=85, x2=95, y1=2, y2=7)])

    result = filter_coherent_row_delimiters(row_delimiters=row_delimiters,
                                            delimiter_group=delimiter_group)

    expected = [Cell(x1=0, x2=100, y1=0, y2=0),
                Cell(x1=0, x2=100, y1=20, y2=20)]

    assert result == expected


def test_correct_delimiter_width():
    row_delimiters = [Cell(x1=0, x2=100, y1=0, y2=0),
                      Cell(x1=0, x2=80, y1=10, y2=10),
                      Cell(x1=30, x2=100, y1=20, y2=20),
                      Cell(x1=0, x2=100, y1=30, y2=30)]

    contours = [Cell(x1=23, x2=34, y1=12, y2=18),
                Cell(x1=86, x2=93, y1=2, y2=9),
                Cell(x1=3, x2=17, y1=18, y2=24)]

    result = correct_delimiter_width(row_delimiters=row_delimiters,
                                     contours=contours)

    expected = [Cell(x1=0, x2=100, y1=0, y2=0),
                Cell(x1=0, x2=100, y1=10, y2=10),
                Cell(x1=17, x2=100, y1=20, y2=20),
                Cell(x1=0, x2=100, y1=30, y2=30)]

    assert result == expected


def test_identify_delimiter_group_rows():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    with open("test_data/contours.json", 'r') as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = identify_delimiter_group_rows(delimiter_group=delimiter_group,
                                           contours=contours)

    assert len(result) == 18
    assert min([d.y1 for d in result]) == 45
    assert max([d.y2 for d in result]) == 1147
    assert min([d.x1 for d in result]) == 53
    assert max([d.x2 for d in result]) == 1277
