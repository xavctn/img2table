# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables import identify_table
from img2table.tables.processing.borderless_tables.model import ColumnGroup, Column, VerticalWS, Whitespace


def test_identify_table():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    column_group = ColumnGroup(columns=[Column(whitespaces=[VerticalWS(ws=Whitespace(cells=[Cell(**col)]))])
                                        for col in data.get('delimiters')],
                               elements=[Cell(**c) for c in data.get('elements')],
                               char_length=4.66)

    with open("test_data/contours.json", 'r') as f:
        contours = [Cell(**el) for el in json.load(f)]

    with open("test_data/rows.json", "r") as f:
        row_delimiters = [Cell(**c) for c in json.load(f)]

    result = identify_table(columns=column_group,
                            row_delimiters=row_delimiters,
                            contours=contours,
                            median_line_sep=16,
                            char_length=4.66)

    assert result.nb_rows == 17
    assert result.nb_columns == 8
    assert (result.x1, result.y1, result.x2, result.y2) == (91, 45, 1235, 1147)
