# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables import identify_column_groups
from img2table.tables.processing.borderless_tables.model import ImageSegment


def test_identify_column_groups():
    with open("test_data/image_segment.json", "r") as f:
        data = json.load(f)
    img_segment = ImageSegment(x1=data.get('x1'),
                               y1=data.get('y1'),
                               x2=data.get('x2'),
                               y2=data.get('y2'),
                               elements=[Cell(**c) for c in data.get('elements')])

    result = identify_column_groups(segment=img_segment,
                                    char_length=7.24)

    assert len(result) == 1
    assert len(result[0].delimiters) == 9
    assert (result[0].x1, result[0].y1, result[0].x2, result[0].y2) == (53, 45, 1277, 1147)
    assert len(result[0].elements) == 164
