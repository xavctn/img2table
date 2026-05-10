from img2table.tables.types import Cell, Row


def test_row() -> None:
    row = Row(cells=[Cell(x1=0, x2=20, y1=0, y2=20), Cell(x1=20, x2=40, y1=0, y2=20)])

    assert row.x1 == 0
    assert row.y1 == 0
    assert row.x2 == 40
    assert row.y2 == 20
    assert row.nb_columns == 2
    assert row.v_consistent


def test_add_cells() -> None:
    row = Row(cells=[Cell(x1=0, x2=20, y1=0, y2=20), Cell(x1=20, x2=40, y1=0, y2=20)])

    row.add_cells(cells=[Cell(x1=40, x2=60, y1=0, y2=20)])

    assert row.nb_columns == 3
    assert row.x2 == 60
