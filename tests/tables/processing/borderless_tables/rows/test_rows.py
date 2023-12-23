# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables import detect_delimiter_group_rows
from img2table.tables.processing.borderless_tables.model import DelimiterGroup


def test_detect_delimiter_group_rows():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    with open("test_data/contours.json", 'r') as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = detect_delimiter_group_rows(delimiter_group=delimiter_group,
                                         contours=contours)

    with open("test_data/rows.json", "r") as f:
        expected = [Cell(**c) for c in json.load(f)]

    assert set(result) == set(expected)
