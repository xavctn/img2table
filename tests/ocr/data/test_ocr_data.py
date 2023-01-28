# coding: utf-8
import json

import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table


def test_pages():
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    ocr_df_page_0 = ocr_df.page(page_number=0)
    ocr_df_page_1 = ocr_df.page(page_number=1)

    assert isinstance(ocr_df_page_0, OCRDataframe)
    assert isinstance(ocr_df_page_1, OCRDataframe)

    assert not ocr_df_page_0 == ocr_df_page_1
    assert len(ocr_df_page_0.df.collect()) + len(ocr_df_page_1.df.collect()) == len(ocr_df.df.collect())


def test_get_text_cell():
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())
    cell = Cell(x1=200, x2=800, y1=700, y2=850)

    result = ocr_df.get_text_cell(cell=cell,
                                  min_confidence=50,
                                  page_number=0)

    assert result == "Use these data to create\nChecklist for a Data Table."


def test_get_text_table():
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    with open("test_data/table.json", "r") as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    result = ocr_df.get_text_table(table=table,
                                   page_number=0,
                                   min_confidence=50)

    with open("test_data/expected_table.json", "r") as f:
        expected = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    assert result == expected
