import json
from pathlib import Path

from img2table.tables.borderless.layout.layout import (
    ColumnDelimiters,
    VerticalWhitespace,
    create_all_layout_areas,
    identify_column_regions,
    identify_layout,
    identify_vertical_delimiters,
)
from img2table.tables.types import Cell, Line


def test_identify_vertical_delimiters() -> None:
    with Path("test_data/contours.json").open() as f:
        contours = [Cell(**row) for row in json.load(f)]
    with Path("test_data/lines.json").open() as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get("h_lines") + data.get("v_lines")]

    result = identify_vertical_delimiters(contours=contours, lines=lines, min_width=12, width=793)

    assert result == [VerticalWhitespace(x1=389, y1=133, x2=404, y2=1054)]


def test_identify_column_regions() -> None:
    result = identify_column_regions(
        vertical_ws=[VerticalWhitespace(x1=389, y1=133, x2=404, y2=1054)],
        x_min=56,
        x_max=737,
        width=793,
    )

    assert all(col_del.relevant for col_del in result)
    assert result == [
        ColumnDelimiters(
            x_min=56,
            x_max=737,
            width=793,
            vertical_ws=[VerticalWhitespace(x1=389, y1=133, x2=404, y2=1054)],
        )
    ]


def test_create_all_layout_areas() -> None:
    with Path("test_data/contours.json").open() as f:
        contours = [Cell(**row) for row in json.load(f)]

    result = create_all_layout_areas(
        column_dels=[
            ColumnDelimiters(
                x_min=56,
                x_max=737,
                width=793,
                vertical_ws=[VerticalWhitespace(x1=389, y1=133, x2=404, y2=1054)],
            )
        ],
        contours=contours,
        y_min=105,
        y_max=1054,
        width=793,
    )

    assert len(result) == 3

    region_1, region_2, region_3 = result
    assert (region_1.x1, region_1.x2, region_1.y1, region_1.y2) == (0, 793, 105, 133)
    assert (region_2.x1, region_2.x2, region_2.y1, region_2.y2) == (0, 396, 133, 1054)
    assert (region_3.x1, region_3.x2, region_3.y1, region_3.y2) == (396, 793, 133, 1054)


def test_identify_layout() -> None:
    with Path("test_data/contours.json").open() as f:
        contours = [Cell(**row) for row in json.load(f)]
    with Path("test_data/lines.json").open() as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get("h_lines") + data.get("v_lines")]

    result = identify_layout(contours=contours, lines=lines, min_width=12, width=793)

    assert len(result) == 3

    region_1, region_2, region_3 = result
    assert (region_1.x1, region_1.x2, region_1.y1, region_1.y2) == (0, 793, 105, 133)
    assert (region_2.x1, region_2.x2, region_2.y1, region_2.y2) == (0, 396, 133, 1054)
    assert (region_3.x1, region_3.x2, region_3.y1, region_3.y2) == (396, 793, 133, 1054)
