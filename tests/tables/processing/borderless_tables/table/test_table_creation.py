# coding: utf-8
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.model import LineGroup, TableLine
from img2table.tables.processing.borderless_tables.table.table_creation import reprocess_line_group, get_table, \
    create_table


def test_reprocess_line_group():
    line_group = LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=100, y1=0, y2=10)]),
                                  TableLine(cells=[Cell(x1=0, x2=100, y1=20, y2=30)]),
                                  TableLine(cells=[Cell(x1=0, x2=100, y1=25, y2=35)]),
                                  TableLine(cells=[Cell(x1=0, x2=100, y1=40, y2=50)])])

    column_delimiters = [Cell(x1=30, x2=35, y1=15, y2=50), Cell(x1=55, x2=60, y1=15, y2=50)]

    result = reprocess_line_group(line_group=line_group,
                                  column_delimiters=column_delimiters)

    expected = [Cell(x1=0, x2=100, y1=20, y2=35), Cell(x1=0, x2=100, y1=40, y2=50)]

    assert result == expected


def test_get_table():
    lines = [Cell(x1=0, x2=100, y1=20, y2=35), Cell(x1=0, x2=100, y1=40, y2=50)]
    column_delimiters = [Cell(x1=30, x2=35, y1=15, y2=50), Cell(x1=55, x2=60, y1=15, y2=50)]

    result = get_table(lines=lines,
                       column_delimiters=column_delimiters)

    expected = Table(rows=[Row(cells=[Cell(x1=0, x2=32, y1=20, y2=38),
                                      Cell(x1=32, x2=58, y1=20, y2=38),
                                      Cell(x1=58, x2=100, y1=20, y2=38)]),
                           Row(cells=[Cell(x1=0, x2=32, y1=38, y2=50),
                                      Cell(x1=32, x2=58, y1=38, y2=50),
                                      Cell(x1=58, x2=100, y1=38, y2=50)])
                           ])

    assert result == expected


def test_create_table():
    line_group = LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=100, y1=0, y2=10)]),
                                  TableLine(cells=[Cell(x1=0, x2=100, y1=20, y2=30)]),
                                  TableLine(cells=[Cell(x1=0, x2=100, y1=25, y2=35)]),
                                  TableLine(cells=[Cell(x1=0, x2=100, y1=40, y2=50)])])

    column_delimiters = [Cell(x1=30, x2=35, y1=15, y2=50), Cell(x1=55, x2=60, y1=15, y2=50)]

    result = create_table(line_group=line_group,
                          column_delimiters=column_delimiters)

    expected = Table(rows=[Row(cells=[Cell(x1=0, x2=32, y1=20, y2=38),
                                      Cell(x1=32, x2=58, y1=20, y2=38),
                                      Cell(x1=58, x2=100, y1=20, y2=38)]),
                           Row(cells=[Cell(x1=0, x2=32, y1=38, y2=50),
                                      Cell(x1=32, x2=58, y1=38, y2=50),
                                      Cell(x1=58, x2=100, y1=38, y2=50)])
                           ])

    assert result == expected
