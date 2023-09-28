# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow
from img2table.tables.processing.borderless_tables.table.table_creation import get_table, \
    get_coherent_columns_dimensions


def test_get_coherent_columns_dimensions():
    columns = DelimiterGroup(delimiters=[Cell(x1=0, x2=0, y1=0, y2=100),
                                         Cell(x1=100, x2=100, y1=0, y2=100),
                                         Cell(x1=200, x2=200, y1=0, y2=100),
                                         Cell(x1=300, x2=300, y1=0, y2=100)])

    table_rows = [TableRow(cells=[Cell(x1=10, x2=130, y1=10, y2=20),
                                  Cell(x1=120, x2=180, y1=10, y2=20)])]

    result = get_coherent_columns_dimensions(columns=columns,
                                             table_rows=table_rows)

    expected = DelimiterGroup(delimiters=[Cell(x1=0, x2=0, y1=0, y2=100),
                                          Cell(x1=100, x2=100, y1=0, y2=100),
                                          Cell(x1=200, x2=200, y1=0, y2=100)])

    assert result == expected


def test_get_table():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    with open("test_data/contours.json", 'r') as f:
        contours = [Cell(**el) for el in json.load(f)]

    with open("test_data/rows.json", "r") as f:
        table_rows = [TableRow(cells=[Cell(**c) for c in row]) for row in json.load(f)]

    result = get_table(columns=delimiter_group,
                       table_rows=table_rows,
                       contours=contours)

    assert result.nb_rows == 16
    assert result.nb_columns == 8
    assert (result.x1, result.y1, result.x2, result.y2) == (93, 45, 1233, 1060)
