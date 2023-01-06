# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.tables import get_tables


def test_get_tables():
    with open("test_data/cells.json", 'r') as f:
        cells = [Cell(**el) for el in json.load(f)]

    result = get_tables(cells=cells)

    with open("test_data/expected.json", "r") as f:
        expected = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                    for tb in json.load(f)]

    assert result == expected

