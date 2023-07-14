# coding: utf-8
import json
from io import BytesIO

from xlsxwriter import Workbook

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.extraction import create_all_rectangles, CellPosition, TableCell, BBox
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table


def test_create_all_rectangles():
    c = TableCell(bbox=BBox(x1=0, y1=0, x2=0, y2=0), value="Test")
    cell_positions = [CellPosition(cell=c, row=0, col=0), CellPosition(cell=c, row=1, col=0),
                      CellPosition(cell=c, row=2, col=0), CellPosition(cell=c, row=3, col=0),
                      CellPosition(cell=c, row=0, col=1), CellPosition(cell=c, row=1, col=1),
                      CellPosition(cell=c, row=2, col=1), CellPosition(cell=c, row=3, col=1),
                      CellPosition(cell=c, row=2, col=2), CellPosition(cell=c, row=3, col=2),
                      CellPosition(cell=c, row=2, col=3), CellPosition(cell=c, row=3, col=3),
                      ]

    result = create_all_rectangles(cell_positions=cell_positions)

    assert result == [(0, 0, 1, 3), (2, 2, 3, 3)]


def test_extracted_table_worksheet():
    with open("test_data/expected_tables.json", "r") as f:
        tables = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                  for tb in json.load(f)]

    wb = Workbook(BytesIO())
    for table in tables:
        ws = wb.add_worksheet()
        extracted_table = table.extracted_table
        extracted_table._to_worksheet(sheet=ws)

        assert ws.dim_colmax + 1 == table.nb_columns
        assert ws.dim_rowmax + 1 == table.nb_rows

        str_map = {v: k for k, v in ws.str_table.string_table.items()}
        ws_values = sorted([str_map.get(c.string) for row in ws.table.values() for c in row.values()])
        table_values = sorted(set([c.value for row in extracted_table.content.values()
                                   for c in row]))
        assert ws_values == table_values

    wb.close()
