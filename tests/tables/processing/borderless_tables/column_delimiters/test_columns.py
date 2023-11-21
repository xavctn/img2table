# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.column_delimiters.columns import get_coherent_ws_height, \
    corresponding_whitespaces, identify_missing_vertical_whitespaces, distance_to_elements, \
    get_coherent_whitespace_position, get_column_whitespaces
from img2table.tables.processing.borderless_tables.model import TableSegment, ImageSegment, DelimiterGroup


def test_get_coherent_ws_height():
    vertical_ws = [Cell(x1=0, x2=100, y1=0, y2=100), Cell(x1=200, x2=300, y1=0, y2=100)]
    unused_ws = [Cell(x1=50, x2=60, y1=12, y2=64)]
    elements = [Cell(x1=23, x2=36, y1=18, y2=76), Cell(x1=57, x2=78, y1=55, y2=82)]

    new_v_ws, new_u_ws = get_coherent_ws_height(vertical_ws=vertical_ws,
                                                unused_ws=unused_ws,
                                                elements=elements)

    assert new_v_ws == [Cell(x1=0, x2=100, y1=18, y2=82), Cell(x1=200, x2=300, y1=18, y2=82)]
    assert new_u_ws == [Cell(x1=50, x2=60, y1=18, y2=64)]


def test_corresponding_whitespaces():
    ws_1 = Cell(x1=50, x2=60, y1=12, y2=64)
    ws_2 = Cell(x1=62, x2=66, y1=68, y2=86)
    ws_3 = Cell(x1=50, x2=60, y1=130, y2=156)

    assert corresponding_whitespaces(ws_1=ws_1, ws_2=ws_2, char_length=10, median_line_sep=20)
    assert not corresponding_whitespaces(ws_1=ws_1, ws_2=ws_3, char_length=10, median_line_sep=20)
    assert not corresponding_whitespaces(ws_1=ws_2, ws_2=ws_3, char_length=10, median_line_sep=20)


def test_identify_missing_vertical_whitespaces():
    unused_ws = [Cell(x1=50, x2=60, y1=12, y2=64), Cell(x1=62, x2=66, y1=68, y2=86),
                 Cell(x1=50, x2=60, y1=130, y2=156), Cell(x1=212, x2=313, y1=28, y2=212),
                 Cell(x1=330, x2=366, y1=28, y2=36)]

    result = identify_missing_vertical_whitespaces(unused_ws=unused_ws,
                                                   char_length=10,
                                                   median_line_sep=20,
                                                   ref_height=150)

    expected = [Cell(x1=212, y1=28, x2=313, y2=212)]

    assert result == expected


def test_distance_to_elements():
    elements = [Cell(x1=23, x2=36, y1=18, y2=76), Cell(x1=57, x2=78, y1=55, y2=82)]

    assert distance_to_elements(x=63, elements=elements) == (1, 7)
    assert distance_to_elements(x=42, elements=elements) == (2, 7)


def test_get_coherent_whitespace_position():
    elements = [Cell(x1=23, x2=36, y1=18, y2=76), Cell(x1=57, x2=78, y1=55, y2=82)]

    ws_1 = Cell(x1=0, x2=12, y1=0, y2=100)
    assert get_coherent_whitespace_position(ws=ws_1, elements=elements) == Cell(x1=23, x2=23, y1=0, y2=100)

    ws_2 = Cell(x1=78, x2=100, y1=0, y2=100)
    assert get_coherent_whitespace_position(ws=ws_2, elements=elements) == Cell(x1=78, x2=78, y1=0, y2=100)

    ws_3 = Cell(x1=42, x2=55, y1=0, y2=100)
    assert get_coherent_whitespace_position(ws=ws_3, elements=elements) == Cell(x1=46, x2=46, y1=0, y2=100)

    ws_4 = Cell(x1=32, x2=60, y1=0, y2=100)
    assert get_coherent_whitespace_position(ws=ws_4, elements=elements) == Cell(x1=37, x2=37, y1=0, y2=100)


def test_get_column_whitespaces():
    vertical_ws = [Cell(x1=7, y1=0, x2=21, y2=660),
                   Cell(x1=270, y1=69, x2=372, y2=626),
                   Cell(x1=1659, y1=69, x2=1758, y2=626),
                   Cell(x1=1845, y1=0, x2=1859, y2=660)]

    with open("test_data/table_segment.json", "r") as f:
        data = json.load(f)

    table_segment = TableSegment(table_areas=[
        ImageSegment(x1=tb.get('x1'), y1=tb.get('y1'), x2=tb.get('x2'), y2=tb.get('y2'),
                     elements=[Cell(**el) for el in tb.get('elements')],
                     whitespaces=[Cell(**el) for el in tb.get('whitespaces')],
                     position=tb.get('position'))
        for tb in data.get("table_areas")
    ])

    result = get_column_whitespaces(vertical_ws=vertical_ws,
                                    unused_ws=[],
                                    table_segment=table_segment,
                                    char_length=14,
                                    median_line_sep=85.75)

    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
        expected = DelimiterGroup(delimiters=[Cell(**d) for d in data.get('delimiters')],
                                  elements=[Cell(**el) for el in data.get('elements')])

    assert result == expected
