import json
from pathlib import Path

import pytest

from img2table._validation import ValidationError
from img2table.ocr.data import OCRData
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from tests.ocr_data_utils import read_ocr_data


def test_validators() -> None:
    with pytest.raises(ValidationError):
        OCRData(records={})  # ty:ignore[invalid-argument-type]


def test_get_text_cell() -> None:
    ocr_data = read_ocr_data("test_data/ocr.csv")
    cell = Cell(x1=200, x2=800, y1=700, y2=850)

    result = ocr_data.get_text_cell(cell=cell, min_confidence=50)

    assert result == "Use these data to create\nChecklist for a Data Table."


def test_get_text_table() -> None:
    ocr_data = read_ocr_data("test_data/ocr.csv")

    with Path("test_data/table.json").open() as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    result = ocr_data.get_text_table(table=table, min_confidence=50)

    with Path("test_data/expected_table.json").open() as f:
        expected = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    assert result == expected
