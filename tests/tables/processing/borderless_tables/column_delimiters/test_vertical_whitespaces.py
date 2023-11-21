# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.column_delimiters.vertical_whitespaces import \
    get_vertical_whitespaces, VertWS, deduplicate_whitespaces
from img2table.tables.processing.borderless_tables.model import ImageSegment, TableSegment


def test_deduplicate_whitespaces():
    v_ws = [VertWS(x1=0, x2=10, whitespaces=[Cell(x1=0, x2=10, y1=0, y2=100)]),
            VertWS(x1=10, x2=12, whitespaces=[Cell(x1=10, x2=12, y1=0, y2=80)]),
            VertWS(x1=40, x2=50, whitespaces=[Cell(x1=40, x2=50, y1=0, y2=100)]),
            VertWS(x1=60, x2=70, whitespaces=[Cell(x1=60, x2=70, y1=0, y2=110)])]

    elements = [Cell(x1=20, x2=30, y1=10, y2=50), Cell(x1=80, x2=100, y1=10, y2=50)]

    result = deduplicate_whitespaces(vertical_whitespaces=v_ws, elements=elements)

    expected = [VertWS(x1=0, x2=10, whitespaces=[Cell(x1=0, x2=10, y1=0, y2=100)]),
                VertWS(x1=60, x2=70, whitespaces=[Cell(x1=60, x2=70, y1=0, y2=110)])]

    assert result == expected


def test_get_vertical_whitespaces():
    with open("test_data/table_segment.json", "r") as f:
        data = json.load(f)

    table_segment = TableSegment(table_areas=[
        ImageSegment(x1=tb.get('x1'), y1=tb.get('y1'), x2=tb.get('x2'), y2=tb.get('y2'),
                     elements=[Cell(**el) for el in tb.get('elements')],
                     whitespaces=[Cell(**el) for el in tb.get('whitespaces')],
                     position=tb.get('position'))
        for tb in data.get("table_areas")
    ])

    vertical_ws, unused_ws = get_vertical_whitespaces(table_segment=table_segment)

    assert len(vertical_ws) == 4
    assert len(unused_ws) == 0

