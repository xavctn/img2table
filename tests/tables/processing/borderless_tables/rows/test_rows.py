# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables import detect_delimiter_group_rows
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow


def test_detect_delimiter_group_rows():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    result = detect_delimiter_group_rows(delimiter_group=delimiter_group)

    with open("test_data/rows.json", "r") as f:
        expected = sorted([TableRow(cells=[Cell(**c) for c in row]) for row in json.load(f)], key=lambda r: r.y1)[:-1]

    assert set(result) == set(expected)
