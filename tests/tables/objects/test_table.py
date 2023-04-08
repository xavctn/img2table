# coding: utf-8
import json

import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table


def test_remove_rows():
    table = Table(rows=[Row(cells=[Cell(x1=0, x2=100, y1=0, y2=10)]),
                        Row(cells=[Cell(x1=0, x2=100, y1=10, y2=20)]),
                        Row(cells=[Cell(x1=0, x2=100, y1=20, y2=30)])])
    table.remove_rows(row_ids=[1])

    expected = Table(rows=[Row(cells=[Cell(x1=0, x2=100, y1=0, y2=15)]),
                           Row(cells=[Cell(x1=0, x2=100, y1=15, y2=30)])])

    assert table == expected


def test_remove_columns():
    table = Table(rows=[Row(cells=[Cell(x1=0, x2=100, y1=0, y2=10),
                                   Cell(x1=100, x2=200, y1=0, y2=10),
                                   Cell(x1=200, x2=300, y1=0, y2=10)]),
                        Row(cells=[Cell(x1=0, x2=100, y1=10, y2=20),
                                   Cell(x1=100, x2=200, y1=10, y2=20),
                                   Cell(x1=200, x2=300, y1=10, y2=20)]),
                        ])

    table.remove_columns(col_ids=[1])

    expected = Table(rows=[Row(cells=[Cell(x1=0, x2=150, y1=0, y2=10),
                                      Cell(x1=150, x2=300, y1=0, y2=10)]),
                           Row(cells=[Cell(x1=0, x2=150, y1=10, y2=20),
                                      Cell(x1=150, x2=300, y1=10, y2=20)])
                           ])

    assert table == expected


def test_table():
    with open("test_data/tables.json", "r") as f:
        tables = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                  for tb in json.load(f)]

    assert tables[0].nb_columns == 3
    assert tables[0].nb_rows == 6
    assert tables[0].bbox() == (35, 20, 770, 326)

    assert tables[1].nb_columns == 2
    assert tables[1].nb_rows == 2
    assert tables[1].bbox() == (961, 21, 1154, 123)


def test_get_table_content():
    with open("test_data/tables.json", "r") as f:
        tables = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                  for tb in json.load(f)]

    # Load OCR
    ocr_df = OCRDataframe(pl.read_csv("test_data/ocr.csv", separator=";", encoding="utf-8").lazy())

    result = [table.get_content(ocr_df=ocr_df, min_confidence=50) for table in tables]

    with open("test_data/expected_tables.json", "r") as f:
        expected = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                    for tb in json.load(f)]

    assert result == expected
