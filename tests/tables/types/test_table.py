import json
from pathlib import Path

from img2table.tables.types import Cell, Row, Table
from tests.ocr_data_utils import read_ocr_data


def test_remove_rows() -> None:
    table = Table(
        rows=[
            Row(cells=[Cell(x1=0, x2=100, y1=0, y2=10)]),
            Row(cells=[Cell(x1=0, x2=100, y1=10, y2=20)]),
            Row(cells=[Cell(x1=0, x2=100, y1=20, y2=30)]),
        ]
    )
    table.remove_rows(row_ids=[1])

    expected = Table(
        rows=[
            Row(cells=[Cell(x1=0, x2=100, y1=0, y2=15)]),
            Row(cells=[Cell(x1=0, x2=100, y1=15, y2=30)]),
        ]
    )

    assert table == expected


def test_remove_columns() -> None:
    table = Table(
        rows=[
            Row(
                cells=[
                    Cell(x1=0, x2=100, y1=0, y2=10),
                    Cell(x1=100, x2=200, y1=0, y2=10),
                    Cell(x1=200, x2=300, y1=0, y2=10),
                ]
            ),
            Row(
                cells=[
                    Cell(x1=0, x2=100, y1=10, y2=20),
                    Cell(x1=100, x2=200, y1=10, y2=20),
                    Cell(x1=200, x2=300, y1=10, y2=20),
                ]
            ),
        ]
    )

    table.remove_columns(col_ids=[1])

    expected = Table(
        rows=[
            Row(cells=[Cell(x1=0, x2=150, y1=0, y2=10), Cell(x1=150, x2=300, y1=0, y2=10)]),
            Row(cells=[Cell(x1=0, x2=150, y1=10, y2=20), Cell(x1=150, x2=300, y1=10, y2=20)]),
        ]
    )

    assert table == expected


def test_table() -> None:
    with Path("test_data/tables.json").open() as f:
        tables = [
            Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb]) for tb in json.load(f)
        ]

    assert tables[0].nb_columns == 3
    assert tables[0].nb_rows == 6
    assert (tables[0].x1, tables[0].y1, tables[0].x2, tables[0].y2) == (35, 20, 770, 326)

    assert tables[1].nb_columns == 2
    assert tables[1].nb_rows == 2
    assert (tables[1].x1, tables[1].y1, tables[1].x2, tables[1].y2) == (961, 21, 1154, 123)


def test_get_table_content() -> None:
    with Path("test_data/tables.json").open() as f:
        tables = [
            Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb]) for tb in json.load(f)
        ]

    # Load OCR
    ocr_data = read_ocr_data("test_data/ocr.csv")

    result = [table.get_content(ocr_data=ocr_data, min_confidence=50) for table in tables]

    with Path("test_data/expected_tables.json").open() as f:
        expected = [
            Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb]) for tb in json.load(f)
        ]

    assert result == expected
