# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.tables.merged_cells import handle_vertical_merged_cells, \
    handle_horizontal_merged_cells, handle_merged_cells


def test_handle_vertical_merged_cells():
    with open("test_data/tables_from_cells.json", "r") as f:
        tables = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                  for tb in json.load(f)]

    # Get row for tests
    row = tables[0].items[2]

    result = handle_vertical_merged_cells(row=row)

    with open("test_data/vertical_merged_cells.json", 'r') as f:
        expected = [Row(cells=[Cell(**el) for el in row]) for row in json.load(f)]

    assert result == expected


def test_handle_horizontal_merged_cells():
    with open("test_data/tables_from_cells.json", "r") as f:
        tables = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                  for tb in json.load(f)]

    result = handle_horizontal_merged_cells(table=tables[0])

    with open("test_data/horizontal_merged_cells.json", 'r') as f:
        expected = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    assert result == expected


def test_handle_merged_cells():
    with open("test_data/tables_from_cells.json", "r") as f:
        tables = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                  for tb in json.load(f)]

    result = [handle_merged_cells(table=table) for table in tables]

    with open("test_data/expected.json", "r") as f:
        expected = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                    for tb in json.load(f)]

    assert result == expected
