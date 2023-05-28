# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables import identify_table
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow


def test_identify_table():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    with open("test_data/rows.json", "r") as f:
        table_rows = [TableRow(cells=[Cell(**c) for c in row]) for row in json.load(f)]

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = identify_table(columns=delimiter_group,
                            table_rows=table_rows,
                            lines=lines)

    assert result.nb_rows == 16
    assert result.nb_columns == 8
    assert (result.x1, result.y1, result.x2, result.y2) == (93, 45, 1233, 1060)
