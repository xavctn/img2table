# coding: utf-8
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row


def test_row():
    row = Row(cells=[Cell(x1=0, x2=20, y1=0, y2=20), Cell(x1=20, x2=40, y1=0, y2=20)])

    assert row.x1 == 0
    assert row.y1 == 0
    assert row.x2 == 40
    assert row.y2 == 20
    assert row.nb_columns == 2
    assert row.v_consistent


def test_add_cells():
    row = Row(cells=[Cell(x1=0, x2=20, y1=0, y2=20), Cell(x1=20, x2=40, y1=0, y2=20)])

    row.add_cells(cells=Cell(x1=40, x2=60, y1=0, y2=20))

    assert row.nb_columns == 3
    assert row.x2 == 60


def test_split_in_rows():
    row = Row(cells=[Cell(x1=0, x2=20, y1=0, y2=20), Cell(x1=20, x2=40, y1=0, y2=20)])

    rows_splitted = row.split_in_rows(vertical_delimiters=[10, 15])

    expected = [Row(cells=[Cell(x1=0, x2=20, y1=0, y2=10), Cell(x1=20, x2=40, y1=0, y2=10)]),
                Row(cells=[Cell(x1=0, x2=20, y1=10, y2=15), Cell(x1=20, x2=40, y1=10, y2=15)]),
                Row(cells=[Cell(x1=0, x2=20, y1=15, y2=20), Cell(x1=20, x2=40, y1=15, y2=20)])
                ]

    assert rows_splitted == expected
