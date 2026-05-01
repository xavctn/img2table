import json
from io import BytesIO
from pathlib import Path

from xlsxwriter import Workbook

from img2table.tables.extraction import (
    BBox,
    RelativeBBox,
    TableCell,
)
from img2table.tables.extraction._utils import CellPosition, CellSpan, create_all_rectangles
from img2table.tables.types import Cell, Row, Table


def test_create_all_rectangles() -> None:
    c = TableCell(bbox=BBox(x1=0, y1=0, x2=0, y2=0), value="Test")
    cell_positions = [
        CellPosition(cell=c, row=0, col=0),
        CellPosition(cell=c, row=1, col=0),
        CellPosition(cell=c, row=2, col=0),
        CellPosition(cell=c, row=3, col=0),
        CellPosition(cell=c, row=0, col=1),
        CellPosition(cell=c, row=1, col=1),
        CellPosition(cell=c, row=2, col=1),
        CellPosition(cell=c, row=3, col=1),
        CellPosition(cell=c, row=2, col=2),
        CellPosition(cell=c, row=3, col=2),
        CellPosition(cell=c, row=2, col=3),
        CellPosition(cell=c, row=3, col=3),
    ]

    result = create_all_rectangles(cell_positions=cell_positions)

    assert result == [
        CellSpan(top_row=0, bottom_row=3, col_left=0, col_right=1, value="Test"),
        CellSpan(top_row=2, bottom_row=3, col_left=2, col_right=3, value="Test"),
    ]


def test_bbox_relative() -> None:
    bbox = BBox(x1=25, y1=20, x2=75, y2=80, image_width=100, image_height=200)

    assert bbox.relative == RelativeBBox(x1=0.25, y1=0.1, x2=0.75, y2=0.4)


def test_table_html() -> None:
    with Path("test_data/tables.json").open() as f:
        table = [
            Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb]) for tb in json.load(f)
        ].pop()

    with Path("test_data/table.html").open() as f:
        expected = f.read()

    assert table.extracted_table(image_width=100, image_height=100).html == expected


def test_extracted_table_worksheet() -> None:
    with Path("test_data/tables.json").open() as f:
        tables = [
            Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb]) for tb in json.load(f)
        ]

    wb = Workbook(BytesIO())
    for table in tables:
        ws = wb.add_worksheet()
        extracted_table = table.extracted_table(image_width=100, image_height=100)
        extracted_table._to_worksheet(sheet=ws)

        assert ws.dim_colmax + 1 == table.nb_columns  # ty:ignore[unsupported-operator]
        assert ws.dim_rowmax + 1 == table.nb_rows  # ty:ignore[unsupported-operator]

        str_map = {v: k for k, v in ws.str_table.string_table.items()}  # ty:ignore[unresolved-attribute]
        ws_values = sorted(
            [str_map.get(c.string) for row in ws.table.values() for c in row.values()]
        )
        table_values = sorted({c.value for row in extracted_table.content.values() for c in row})
        assert ws_values == table_values

    wb.close()
