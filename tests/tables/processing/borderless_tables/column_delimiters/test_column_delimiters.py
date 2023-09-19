# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables import identify_column_groups
from img2table.tables.processing.borderless_tables.model import ImageSegment, TableSegment, DelimiterGroup


def test_identify_column_groups():
    with open("test_data/table_segment.json", "r") as f:
        data = json.load(f)

    table_segment = TableSegment(table_areas=[
        ImageSegment(x1=tb.get('x1'), y1=tb.get('y1'), x2=tb.get('x2'), y2=tb.get('y2'),
                     elements=[Cell(**el) for el in tb.get('elements')],
                     whitespaces=[Cell(**el) for el in tb.get('whitespaces')],
                     position=tb.get('position'))
        for tb in data.get("table_areas")
    ])

    result = identify_column_groups(table_segment=table_segment,
                                    char_length=14,
                                    median_line_sep=85.75)

    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
        expected = DelimiterGroup(delimiters=[Cell(**d) for d in data.get('delimiters')],
                                  elements=[Cell(**el) for el in data.get('elements')])

    assert result == expected

