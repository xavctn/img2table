import json
from pathlib import Path

import polars as pl
import pytest

from img2table._validation import ValidationError
from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table


def test_validators() -> None:
    with pytest.raises(ValidationError):
        OCRDataframe(df={})  # ty:ignore[invalid-argument-type]


def test_pages() -> None:
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    ocr_df_page_0 = ocr_df.page(page_number=0)
    ocr_df_page_1 = ocr_df.page(page_number=1)

    assert isinstance(ocr_df_page_0, OCRDataframe)
    assert isinstance(ocr_df_page_1, OCRDataframe)

    assert ocr_df_page_0 != ocr_df_page_1
    assert len(ocr_df_page_0.df) + len(ocr_df_page_1.df) == len(ocr_df.df)


def test_get_text_cell() -> None:
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))
    cell = Cell(x1=200, x2=800, y1=700, y2=850)

    result = ocr_df.get_text_cell(cell=cell, min_confidence=50, page_number=0)

    assert (
        result
        == "http://www.landspeed.com/lsrinfo.asp.)\nUse these data to create\nChecklist for a Data Table."
    )


def test_get_text_table() -> None:
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    with Path("test_data/table.json").open() as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    result = ocr_df.get_text_table(table=table, page_number=0, min_confidence=50)

    with Path("test_data/expected_table.json").open() as f:
        expected = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    assert result == expected
