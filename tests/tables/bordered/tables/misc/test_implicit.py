import json
from pathlib import Path

from img2table.tables.bordered.tables.misc.implicit import (
    implicit_columns_lines,
    implicit_content,
    implicit_rows_lines,
)
from img2table.tables.types import Cell, Row, Table


def test_implicit_rows_lines() -> None:
    with Path("test_data/table_implicit.json").open() as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    with Path("test_data/contours_implicit.json").open() as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = implicit_rows_lines(table=table, contours=contours)

    # Check that all created lines have right width
    assert all(line.width == table.width for line in result)

    # Check positions
    assert sorted([line.y1 for line in result]) == [
        682,
        716,
        784,
        817,
        884,
        919,
        986,
        1020,
        1089,
        1121,
        1189,
        1223,
        1292,
        1325,
        1394,
        1427,
        1494,
        1529,
        1597,
        1630,
    ]


def test_implicit_columns_lines() -> None:
    with Path("test_data/table_implicit.json").open() as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    with Path("test_data/contours_implicit.json").open() as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = implicit_columns_lines(table=table, contours=contours, char_length=11)

    # Check that all created lines have right height
    assert all(line.height == table.height for line in result)

    # Check positions
    assert sorted([line.x1 for line in result]) == [395, 605, 725, 809, 886, 1212, 1285, 1396]


def test_implicit_content() -> None:
    with Path("test_data/table_implicit.json").open() as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    with Path("test_data/contours_implicit.json").open() as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = implicit_content(
        table=table, contours=contours, char_length=11, implicit_rows=True, implicit_columns=True
    )

    # Check that 20 more rows have been created
    assert result.nb_rows == table.nb_rows + 20

    # Check that 8 more columns have been created
    assert result.nb_columns == table.nb_columns + 8
