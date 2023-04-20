# coding: utf-8
import json

import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.tables.implicit_rows import handle_implicit_rows_table, \
    handle_implicit_rows


def test_handle_implicit_rows_table():
    img = cv2.imread("test_data/implicit.png", cv2.IMREAD_GRAYSCALE)

    with open("test_data/implicit_table.json", 'r') as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    result = handle_implicit_rows_table(img=img, table=table)

    # Check that 2 more lines have been created
    assert result.nb_rows == table.nb_rows + 2


def test_handle_implicit_rows():
    img = cv2.imread("test_data/implicit.png", cv2.IMREAD_GRAYSCALE)

    with open("test_data/implicit_table.json", 'r') as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    result = handle_implicit_rows(img=img, tables=[table])

    # Check that 2 more lines have been created
    assert result[0].nb_rows == table.nb_rows + 2
