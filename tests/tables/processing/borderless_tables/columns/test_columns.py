# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.columns import get_columns_delimiters, get_relevant_height, \
    identify_columns
from img2table.tables.processing.borderless_tables.model import ImageSegment, TableSegment, DelimiterGroup


def test_get_columns_delimiters():
    with open("test_data/table_segment.json", "r") as f:
        data = json.load(f)

    table_segment = TableSegment(table_areas=[
        ImageSegment(x1=tb.get('x1'), y1=tb.get('y1'), x2=tb.get('x2'), y2=tb.get('y2'),
                     elements=[Cell(**el) for el in tb.get('elements')],
                     whitespaces=[Cell(**el) for el in tb.get('whitespaces')],
                     position=tb.get('position'))
        for tb in data.get("table_areas")
    ])

    result = get_columns_delimiters(table_segment=table_segment,
                                    char_length=14)

    assert result == [Cell(x1=7, y1=0, x2=21, y2=660),
                      Cell(x1=270, y1=69, x2=372, y2=626),
                      Cell(x1=1659, y1=69, x2=1758, y2=626),
                      Cell(x1=1845, y1=0, x2=1859, y2=660)]


def test_get_relevant_height():
    whitespaces = [Cell(x1=7, y1=0, x2=21, y2=660),
                   Cell(x1=270, y1=69, x2=372, y2=626),
                   Cell(x1=1659, y1=69, x2=1758, y2=626),
                   Cell(x1=1845, y1=0, x2=1859, y2=660)]

    with open("test_data/table_segment.json", "r") as f:
        data = json.load(f)
    elements = [Cell(**el) for tb in data.get("table_areas") for el in tb.get('elements')]

    result = get_relevant_height(whitespaces=whitespaces,
                                 elements=elements,
                                 char_length=14)

    assert result == [Cell(x1=7, y1=111, x2=7, y2=626),
                      Cell(x1=321, y1=111, x2=321, y2=626),
                      Cell(x1=1708, y1=111, x2=1708, y2=626),
                      Cell(x1=1859, y1=111, x2=1859, y2=626)]


def test_identify_columns():
    with open("test_data/table_segment.json", "r") as f:
        data = json.load(f)

    table_segment = TableSegment(table_areas=[
        ImageSegment(x1=tb.get('x1'), y1=tb.get('y1'), x2=tb.get('x2'), y2=tb.get('y2'),
                     elements=[Cell(**el) for el in tb.get('elements')],
                     whitespaces=[Cell(**el) for el in tb.get('whitespaces')],
                     position=tb.get('position'))
        for tb in data.get("table_areas")
    ])

    result = identify_columns(table_segment=table_segment,
                              char_length=14)

    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
        expected = DelimiterGroup(delimiters=[Cell(**d) for d in data.get('delimiters')],
                                  elements=[Cell(**el) for el in data.get('elements')])

    assert result == expected

