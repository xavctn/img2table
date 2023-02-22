# coding: utf-8

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.table_creation.columns import get_column_delimiters


def test_get_column_delimiters():
    tb_clusters = [
        [Cell(x1=10, x2=20, y1=10, y2=20), Cell(x1=10, x2=20, y1=20, y2=30)],
        [Cell(x1=40, x2=60, y1=10, y2=20), Cell(x1=40, x2=60, y1=20, y2=30)],
        [Cell(x1=80, x2=100, y1=10, y2=20), Cell(x1=80, x2=100, y1=20, y2=30)],
    ]
    result = get_column_delimiters(tb_clusters=tb_clusters, margin=5)

    assert result == ([5, 30, 70, 105],
                      [[Cell(x1=10, y1=10, x2=20, y2=20), Cell(x1=10, y1=20, x2=20, y2=30)],
                       [Cell(x1=40, y1=10, x2=60, y2=20), Cell(x1=40, y1=20, x2=60, y2=30)],
                       [Cell(x1=80, y1=10, x2=100, y2=20), Cell(x1=80, y1=20, x2=100, y2=30)]])
