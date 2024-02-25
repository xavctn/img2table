# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.layout.column_segments import Rectangle, identify_remaining_segments, \
    get_vertical_ws, is_column_section, identify_column_groups, get_column_group_segments, get_segments_from_columns, \
    segment_image_columns
from img2table.tables.processing.borderless_tables.model import ImageSegment, Whitespace


def test_identify_remaining_segments():
    searched_rectangle = Rectangle(x1=0, y1=0, x2=100, y2=100)
    existing_segments = [Cell(x1=0, y1=25, x2=35, y2=40),
                         Cell(x1=59, y1=37, x2=78, y2=49)]

    result = identify_remaining_segments(searched_rectangle=searched_rectangle,
                                         existing_segments=existing_segments)

    expected = [Cell(x1=0, y1=49, x2=100, y2=100),
                Cell(x1=0, y1=0, x2=100, y2=25),
                Cell(x1=35, y1=25, x2=100, y2=37),
                Cell(x1=0, y1=40, x2=59, y2=49),
                Cell(x1=78, y1=37, x2=100, y2=49)]

    assert result == expected


def test_get_vertical_ws():
    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    with open("test_data/elements.json", "r") as f:
        elements = [Cell(**el) for el in json.load(f)]

    image_segment = ImageSegment(x1=0, y1=49, x2=768, y2=967,
                                 elements=elements)

    result = get_vertical_ws(image_segment=image_segment,
                             char_length=5.04,
                             lines=lines)

    expected = [Whitespace(cells=[Cell(x1=0, y1=105, x2=56, y2=1055)]),
                Whitespace(cells=[Cell(x1=389, y1=117, x2=404, y2=1055)]),
                Whitespace(cells=[Cell(x1=737, y1=105, x2=768, y2=1055)])]

    assert result == expected


def test_is_column_section():
    ws_gp_1 = [Cell(x1=0, x2=10, y1=100, y2=300),
               Cell(x1=148, x2=153, y1=78, y2=292),
               Cell(x1=297, x2=312, y1=113, y2=302)]

    assert is_column_section(ws_group=ws_gp_1)
    assert not is_column_section(ws_group=ws_gp_1 + ws_gp_1)

    ws_gp_2 = [Cell(x1=0, x2=10, y1=100, y2=300),
               Cell(x1=148, x2=153, y1=78, y2=292),
               Cell(x1=397, x2=412, y1=113, y2=302)]

    assert not is_column_section(ws_group=ws_gp_2)


def test_identify_column_groups():
    with open("test_data/elements.json", "r") as f:
        elements = [Cell(**el) for el in json.load(f)]

    image_segment = ImageSegment(x1=0, y1=49, x2=768, y2=967,
                                 elements=elements)

    vertical_ws = [Cell(x1=0, y1=49, x2=51, y2=967),
                   Cell(x1=398, y1=64, x2=405, y2=967),
                   Cell(x1=732, y1=49, x2=768, y2=967)]

    result = identify_column_groups(image_segment=image_segment,
                                    vertical_ws=vertical_ws)

    expected = [[Cell(x1=398, y1=64, x2=405, y2=967),
                 Cell(x1=0, y1=49, x2=51, y2=967),
                 Cell(x1=732, y1=49, x2=768, y2=967)]]

    assert result == expected


def test_get_column_group_segments():
    col_gp = [Cell(x1=0, x2=10, y1=0, y2=100),
              Cell(x1=30, x2=40, y1=30, y2=100),
              Cell(x1=60, x2=70, y1=0, y2=100),
              Cell(x1=90, x2=100, y1=0, y2=100)]

    result = get_column_group_segments(col_group=col_gp)

    expected = [ImageSegment(x1=5, y1=30, x2=35, y2=100),
                ImageSegment(x1=35, y1=30, x2=65, y2=100),
                ImageSegment(x1=65, y1=0, x2=95, y2=100),
                ImageSegment(x1=5, y1=0, x2=65, y2=30)]

    assert result == expected


def test_get_segments_from_columns():
    with open("test_data/elements.json", "r") as f:
        elements = [Cell(**el) for el in json.load(f)]

    image_segment = ImageSegment(x1=0, y1=49, x2=768, y2=967,
                                 elements=elements)

    col_gps = [[Cell(x1=0, x2=10, y1=0, y2=100),
                Cell(x1=30, x2=40, y1=30, y2=100),
                Cell(x1=60, x2=70, y1=0, y2=100),
                Cell(x1=90, x2=100, y1=0, y2=100)]]

    result = get_segments_from_columns(image_segment=image_segment,
                                       column_groups=col_gps)

    expected = [ImageSegment(x1=5, y1=30, x2=35, y2=100),
                ImageSegment(x1=35, y1=30, x2=65, y2=100),
                ImageSegment(x1=65, y1=0, x2=95, y2=100),
                ImageSegment(x1=5, y1=0, x2=65, y2=30),
                ImageSegment(x1=0, y1=49, x2=768, y2=0),
                ImageSegment(x1=0, y1=100, x2=768, y2=967),
                ImageSegment(x1=0, y1=0, x2=5, y2=100),
                ImageSegment(x1=95, y1=0, x2=768, y2=100)]

    assert result == expected


def test_segment_image_columns():
    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    with open("test_data/elements.json", "r") as f:
        elements = [Cell(**el) for el in json.load(f)]

    image_segment = ImageSegment(x1=0, y1=0, x2=793, y2=1123,
                                 elements=elements)

    result = segment_image_columns(image_segment=image_segment,
                                   char_length=6.0,
                                   lines=lines)

    assert len(result) == 3
