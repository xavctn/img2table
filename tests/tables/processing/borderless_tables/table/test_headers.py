# coding: utf-8
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.table.headers import match_table_elements, check_header_coherency, \
    identify_table_lines, headers_from_lines, process_headers


def test_match_table_elements():
    table = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=0, y2=10),
                                   Cell(x1=10, x2=20, y1=0, y2=10),
                                   Cell(x1=20, x2=30, y1=0, y2=10)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=10, y2=20),
                                   Cell(x1=10, x2=20, y1=10, y2=20),
                                   Cell(x1=20, x2=30, y1=10, y2=20)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=20, y2=30),
                                   Cell(x1=10, x2=20, y1=20, y2=30),
                                   Cell(x1=20, x2=30, y1=20, y2=30)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40),
                                   Cell(x1=10, x2=20, y1=30, y2=40),
                                   Cell(x1=20, x2=30, y1=30, y2=40)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100),
                                   Cell(x1=10, x2=20, y1=40, y2=100),
                                   Cell(x1=20, x2=30, y1=40, y2=100)]),
                        ])

    elements = [Cell(x1=0, x2=10, y1=0, y2=10),
                Cell(x1=10, x2=20, y1=10, y2=20),
                Cell(x1=0, x2=10, y1=20, y2=30), Cell(x1=10, x2=20, y1=20, y2=30), Cell(x1=20, x2=30, y1=20, y2=30),
                Cell(x1=0, x2=10, y1=30, y2=40), Cell(x1=10, x2=20, y1=30, y2=40), Cell(x1=20, x2=30, y1=30, y2=40),
                Cell(x1=0, x2=10, y1=40, y2=100), Cell(x1=20, x2=30, y1=40, y2=100)]

    result = match_table_elements(table=table, elements=elements)

    expected = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=0, y2=10, content=True),
                                      Cell(x1=10, x2=20, y1=0, y2=10, content=False),
                                      Cell(x1=20, x2=30, y1=0, y2=10, content=False)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=10, y2=20, content=False),
                                      Cell(x1=10, x2=20, y1=10, y2=20, content=True),
                                      Cell(x1=20, x2=30, y1=10, y2=20, content=False)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=20, y2=30, content=True),
                                      Cell(x1=10, x2=20, y1=20, y2=30, content=True),
                                      Cell(x1=20, x2=30, y1=20, y2=30, content=True)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40, content=True),
                                      Cell(x1=10, x2=20, y1=30, y2=40, content=True),
                                      Cell(x1=20, x2=30, y1=30, y2=40, content=True)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100, content=True),
                                      Cell(x1=10, x2=20, y1=40, y2=100, content=False),
                                      Cell(x1=20, x2=30, y1=40, y2=100, content=True)]),
                           ])

    assert result == expected


def test_check_header_coherency():
    table = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=0, y2=10),
                                   Cell(x1=10, x2=20, y1=0, y2=10),
                                   Cell(x1=20, x2=30, y1=0, y2=10)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=10, y2=20),
                                   Cell(x1=10, x2=20, y1=10, y2=20),
                                   Cell(x1=20, x2=30, y1=10, y2=20)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=20, y2=30),
                                   Cell(x1=10, x2=20, y1=20, y2=30),
                                   Cell(x1=20, x2=30, y1=20, y2=30)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40),
                                   Cell(x1=10, x2=20, y1=30, y2=40),
                                   Cell(x1=20, x2=30, y1=30, y2=40)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100),
                                   Cell(x1=10, x2=20, y1=40, y2=100),
                                   Cell(x1=20, x2=30, y1=40, y2=100)]),
                        ])

    elements = [Cell(x1=0, x2=10, y1=0, y2=10),
                Cell(x1=10, x2=20, y1=10, y2=20),
                Cell(x1=0, x2=10, y1=20, y2=30), Cell(x1=10, x2=20, y1=20, y2=30), Cell(x1=20, x2=30, y1=20, y2=30),
                Cell(x1=0, x2=10, y1=30, y2=40), Cell(x1=10, x2=20, y1=30, y2=40), Cell(x1=20, x2=30, y1=30, y2=40),
                Cell(x1=0, x2=10, y1=40, y2=100), Cell(x1=20, x2=30, y1=40, y2=100)]

    result = check_header_coherency(table=table, elements=elements)

    expected = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=10, y2=20, content=False),
                                      Cell(x1=10, x2=20, y1=10, y2=20, content=True),
                                      Cell(x1=20, x2=30, y1=10, y2=20, content=False)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=20, y2=30, content=True),
                                      Cell(x1=10, x2=20, y1=20, y2=30, content=True),
                                      Cell(x1=20, x2=30, y1=20, y2=30, content=True)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40, content=True),
                                      Cell(x1=10, x2=20, y1=30, y2=40, content=True),
                                      Cell(x1=20, x2=30, y1=30, y2=40, content=True)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100, content=True),
                                      Cell(x1=10, x2=20, y1=40, y2=100, content=False),
                                      Cell(x1=20, x2=30, y1=40, y2=100, content=True)]),
                           ])

    assert result == expected


def test_identify_table_lines():
    table = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=10, y2=20, content=False),
                                   Cell(x1=10, x2=20, y1=10, y2=20, content=True),
                                   Cell(x1=20, x2=30, y1=10, y2=20, content=False)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=20, y2=30, content=True),
                                   Cell(x1=10, x2=20, y1=20, y2=30, content=True),
                                   Cell(x1=20, x2=30, y1=20, y2=30, content=True)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40, content=True),
                                   Cell(x1=10, x2=20, y1=30, y2=40, content=True),
                                   Cell(x1=20, x2=30, y1=30, y2=40, content=True)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100, content=True),
                                   Cell(x1=10, x2=20, y1=40, y2=100, content=False),
                                   Cell(x1=20, x2=30, y1=40, y2=100, content=True)]),
                        ])

    lines = [Line(x1=0, x2=25, y1=12, y2=12),
             Line(x1=0, x2=11, y1=31, y2=31),
             Line(x1=13, x2=26, y1=29, y2=29),
             Line(x1=13, x2=26, y1=38, y2=38),
             Line(x1=17, x2=17, y1=2, y2=23)]

    result = identify_table_lines(table=table, lines=lines)

    assert result == [10, 30]


def test_headers_from_lines():
    table = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=10, y2=20, content=False),
                                   Cell(x1=10, x2=20, y1=10, y2=20, content=True),
                                   Cell(x1=20, x2=30, y1=10, y2=20, content=False)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=20, y2=30, content=True),
                                   Cell(x1=10, x2=20, y1=20, y2=30, content=True),
                                   Cell(x1=20, x2=30, y1=20, y2=30, content=True)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40, content=True),
                                   Cell(x1=10, x2=20, y1=30, y2=40, content=True),
                                   Cell(x1=20, x2=30, y1=30, y2=40, content=True)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100, content=True),
                                   Cell(x1=10, x2=20, y1=40, y2=100, content=False),
                                   Cell(x1=20, x2=30, y1=40, y2=100, content=True)]),
                        ])

    lines = [Line(x1=0, x2=25, y1=12, y2=12),
             Line(x1=0, x2=11, y1=31, y2=31),
             Line(x1=13, x2=26, y1=29, y2=29),
             Line(x1=13, x2=26, y1=38, y2=38),
             Line(x1=17, x2=17, y1=2, y2=23)]

    result = headers_from_lines(table=table, lines=lines)

    expected = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=10, y2=30),
                                      Cell(x1=10, x2=20, y1=10, y2=30),
                                      Cell(x1=20, x2=30, y1=10, y2=30)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40, content=True),
                                      Cell(x1=10, x2=20, y1=30, y2=40, content=True),
                                      Cell(x1=20, x2=30, y1=30, y2=40, content=True)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100, content=True),
                                      Cell(x1=10, x2=20, y1=40, y2=100, content=False),
                                      Cell(x1=20, x2=30, y1=40, y2=100, content=True)]),
                           ])

    assert result == expected


def test_process_headers():
    table = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=0, y2=10),
                                   Cell(x1=10, x2=20, y1=0, y2=10),
                                   Cell(x1=20, x2=30, y1=0, y2=10)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=10, y2=20),
                                   Cell(x1=10, x2=20, y1=10, y2=20),
                                   Cell(x1=20, x2=30, y1=10, y2=20)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=20, y2=30),
                                   Cell(x1=10, x2=20, y1=20, y2=30),
                                   Cell(x1=20, x2=30, y1=20, y2=30)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40),
                                   Cell(x1=10, x2=20, y1=30, y2=40),
                                   Cell(x1=20, x2=30, y1=30, y2=40)]),
                        Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100),
                                   Cell(x1=10, x2=20, y1=40, y2=100),
                                   Cell(x1=20, x2=30, y1=40, y2=100)]),
                        ])

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

    result = process_headers(table=table,
                             lines=lines,
                             elements=elements)

    expected = Table(rows=[Row(cells=[Cell(x1=0, x2=10, y1=10, y2=30),
                                      Cell(x1=10, x2=20, y1=10, y2=30),
                                      Cell(x1=20, x2=30, y1=10, y2=30)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=30, y2=40, content=True),
                                      Cell(x1=10, x2=20, y1=30, y2=40, content=True),
                                      Cell(x1=20, x2=30, y1=30, y2=40, content=True)]),
                           Row(cells=[Cell(x1=0, x2=10, y1=40, y2=100, content=True),
                                      Cell(x1=10, x2=20, y1=40, y2=100, content=False),
                                      Cell(x1=20, x2=30, y1=40, y2=100, content=True)]),
                           ])

    assert result == expected
