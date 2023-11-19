# coding: utf-8

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.bordered_tables.tables import add_semi_bordered_cells


def test_add_semi_bordered_cells():
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    lines = [Line(x1=50, x2=205, y1=100, y2=100),
             Line(x1=50, x2=205, y1=200, y2=200),
             Line(x1=100, x2=100, y1=30, y2=270),
             Line(x1=200, x2=200, y1=30, y2=270)]

    result = add_semi_bordered_cells(cluster=cluster,
                                     lines=lines,
                                     char_length=5)

    expected = [Cell(x1=100, y1=100, x2=200, y2=200),
                Cell(x1=50, y1=200, x2=100, y2=270),
                Cell(x1=100, y1=30, x2=200, y2=100),
                Cell(x1=50, y1=30, x2=100, y2=100),
                Cell(x1=100, y1=200, x2=200, y2=270),
                Cell(x1=50, y1=100, x2=100, y2=200)]

    assert sorted(result, key=lambda c: c.bbox()) == sorted(expected, key=lambda c: c.bbox())
