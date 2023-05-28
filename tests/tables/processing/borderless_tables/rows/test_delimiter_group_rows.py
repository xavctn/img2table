# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup
from img2table.tables.processing.borderless_tables.rows.delimiter_group_rows import \
    get_delimiter_group_row_separation, identify_rows, identify_delimiter_group_rows


def test_get_delimiter_group_row_separation():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    result = get_delimiter_group_row_separation(delimiter_group=delimiter_group)

    assert result == 66


def test_identify_rows():
    with open("test_data/delimiter_group.json", "r") as f:
        elements = [Cell(**c) for c in json.load(f).get('elements')]

    result = identify_rows(elements=elements,
                           ref_size=22)

    assert len(result) == 17
    assert min([row.y1 for row in result]) == 45
    assert max([row.y2 for row in result]) == 1147
    assert min([row.x1 for row in result]) == 93
    assert max([row.x2 for row in result]) == 1233


def test_identify_delimiter_group_rows():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    result, sep = identify_delimiter_group_rows(delimiter_group=delimiter_group)

    assert sep == 66
    assert len(result) == 17
    assert min([row.y1 for row in result]) == 45
    assert max([row.y2 for row in result]) == 1147
    assert min([row.x1 for row in result]) == 93
    assert max([row.x2 for row in result]) == 1233
