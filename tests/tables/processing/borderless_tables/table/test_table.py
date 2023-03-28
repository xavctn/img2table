# coding: utf-8
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables import identify_table
from img2table.tables.processing.borderless_tables.model import LineGroup, TableLine


def test_identify_table():
    line_group = LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=30, y1=0, y2=10)]),
                                  TableLine(cells=[Cell(x1=0, x2=30, y1=10, y2=20)]),
                                  TableLine(cells=[Cell(x1=0, x2=30, y1=20, y2=30)]),
                                  TableLine(cells=[Cell(x1=0, x2=30, y1=30, y2=40)]),
                                  TableLine(cells=[Cell(x1=0, x2=30, y1=40, y2=100)])])

    column_delimiters = [Cell(x1=5, x2=15, y1=0, y2=100), Cell(x1=15, x2=25, y1=0, y2=100)]

    elements = [Cell(x1=0, x2=10, y1=0, y2=10),
                Cell(x1=10, x2=20, y1=10, y2=20),
                Cell(x1=0, x2=10, y1=20, y2=30), Cell(x1=10, x2=20, y1=20, y2=30), Cell(x1=20, x2=30, y1=20, y2=30),
                Cell(x1=0, x2=10, y1=30, y2=40), Cell(x1=10, x2=20, y1=30, y2=40), Cell(x1=20, x2=30, y1=30, y2=40),
                Cell(x1=0, x2=10, y1=40, y2=100), Cell(x1=20, x2=30, y1=40, y2=100)]

    lines = [Line(x1=0, x2=25, y1=12, y2=12),
             Line(x1=0, x2=11, y1=31, y2=31),
             Line(x1=13, x2=26, y1=29, y2=29),
             Line(x1=13, x2=26, y1=38, y2=38),
             Line(x1=17, x2=17, y1=2, y2=23)]

    result = identify_table(line_group=line_group,
                            column_delimiters=column_delimiters,
                            elements=elements,
                            lines=lines)

    expected = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=10, y2=30),
                                      Cell(x1=10, x2=20, y1=10, y2=30),
                                      Cell(x1=20, x2=30, y1=10, y2=30)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40),
                                      Cell(x1=10, x2=20, y1=30, y2=40),
                                      Cell(x1=20, x2=30, y1=30, y2=40)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100),
                                      Cell(x1=10, x2=20, y1=40, y2=100),
                                      Cell(x1=20, x2=30, y1=40, y2=100)]),
                           ])

    assert result == expected
