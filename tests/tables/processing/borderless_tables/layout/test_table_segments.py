# coding: utf-8

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.layout.table_segments import get_table_areas, coherent_table_areas, \
    table_segment_from_group, get_table_segments
from img2table.tables.processing.borderless_tables.model import ImageSegment, TableSegment, Whitespace


def test_get_table_areas():
    elements = [Cell(x1=10, y1=10, x2=20, y2=20), Cell(x1=30, y1=10, x2=40, y2=20), Cell(x1=50, y1=10, x2=60, y2=20),
                Cell(x1=10, y1=25, x2=20, y2=35), Cell(x1=30, y1=25, x2=40, y2=35), Cell(x1=50, y1=25, x2=60, y2=35),
                Cell(x1=10, y1=40, x2=20, y2=50), Cell(x1=50, y1=40, x2=60, y2=50),
                Cell(x1=10, y1=100, x2=20, y2=110), Cell(x1=30, y1=100, x2=40, y2=110),
                Cell(x1=50, y1=100, x2=60, y2=110),
                Cell(x1=10, y1=200, x2=20, y2=210), Cell(x1=30, y1=200, x2=40, y2=210),
                Cell(x1=50, y1=200, x2=60, y2=210)
                ]
    segment = ImageSegment(x1=0, y1=0, x2=1000, y2=1000, elements=elements)

    result = get_table_areas(segment=segment,
                             char_length=4,
                             median_line_sep=10)

    expected = [ImageSegment(x1=6, y1=10, x2=64, y2=20,
                             elements=[Cell(x1=10, y1=10, x2=20, y2=20),
                                       Cell(x1=30, y1=10, x2=40, y2=20),
                                       Cell(x1=50, y1=10, x2=60, y2=20)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=10, x2=10, y2=20)]),
                                          Whitespace(cells=[Cell(x1=20, y1=10, x2=30, y2=20)]),
                                          Whitespace(cells=[Cell(x1=40, y1=10, x2=50, y2=20)]),
                                          Whitespace(cells=[Cell(x1=60, y1=10, x2=64, y2=20)])],
                             position=1),
                ImageSegment(x1=6, y1=25, x2=64, y2=35,
                             elements=[Cell(x1=10, y1=25, x2=20, y2=35),
                                       Cell(x1=30, y1=25, x2=40, y2=35),
                                       Cell(x1=50, y1=25, x2=60, y2=35)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=25, x2=10, y2=35)]),
                                          Whitespace(cells=[Cell(x1=20, y1=25, x2=30, y2=35)]),
                                          Whitespace(cells=[Cell(x1=40, y1=25, x2=50, y2=35)]),
                                          Whitespace(cells=[Cell(x1=60, y1=25, x2=64, y2=35)])],
                             position=2),
                ImageSegment(x1=6, y1=40, x2=64, y2=50,
                             elements=[Cell(x1=10, y1=40, x2=20, y2=50),
                                       Cell(x1=50, y1=40, x2=60, y2=50)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=40, x2=10, y2=50)]),
                                          Whitespace(cells=[Cell(x1=20, y1=40, x2=50, y2=50)]),
                                          Whitespace(cells=[Cell(x1=60, y1=40, x2=64, y2=50)])],
                             position=3),
                ImageSegment(x1=6, y1=100, x2=64, y2=110,
                             elements=[Cell(x1=10, y1=100, x2=20, y2=110),
                                       Cell(x1=30, y1=100, x2=40, y2=110),
                                       Cell(x1=50, y1=100, x2=60, y2=110)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=100, x2=10, y2=110)]),
                                          Whitespace(cells=[Cell(x1=20, y1=100, x2=30, y2=110)]),
                                          Whitespace(cells=[Cell(x1=40, y1=100, x2=50, y2=110)]),
                                          Whitespace(cells=[Cell(x1=60, y1=100, x2=64, y2=110)])],
                             position=4),
                ImageSegment(x1=6, y1=200, x2=64, y2=210,
                             elements=[Cell(x1=10, y1=200, x2=20, y2=210),
                                       Cell(x1=30, y1=200, x2=40, y2=210),
                                       Cell(x1=50, y1=200, x2=60, y2=210)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=200, x2=10, y2=210)]),
                                          Whitespace(cells=[Cell(x1=20, y1=200, x2=30, y2=210)]),
                                          Whitespace(cells=[Cell(x1=40, y1=200, x2=50, y2=210)]),
                                          Whitespace(cells=[Cell(x1=60, y1=200, x2=64, y2=210)])],
                             position=5)]

    assert result == expected


def test_coherent_table_areas():
    tb_area_1 = ImageSegment(x1=6, y1=10, x2=64, y2=20,
                             elements=[Cell(x1=10, y1=10, x2=20, y2=20),
                                       Cell(x1=30, y1=10, x2=40, y2=20),
                                       Cell(x1=50, y1=10, x2=60, y2=20)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=10, x2=10, y2=20)]),
                                          Whitespace(cells=[Cell(x1=20, y1=10, x2=30, y2=20)]),
                                          Whitespace(cells=[Cell(x1=40, y1=10, x2=50, y2=20)]),
                                          Whitespace(cells=[Cell(x1=60, y1=10, x2=64, y2=20)])],
                             position=1)

    tb_area_2 = ImageSegment(x1=6, y1=25, x2=64, y2=35,
                             elements=[Cell(x1=10, y1=25, x2=20, y2=35),
                                       Cell(x1=30, y1=25, x2=40, y2=35),
                                       Cell(x1=50, y1=25, x2=60, y2=35)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=25, x2=10, y2=35)]),
                                          Whitespace(cells=[Cell(x1=20, y1=25, x2=30, y2=35)]),
                                          Whitespace(cells=[Cell(x1=40, y1=25, x2=50, y2=35)]),
                                          Whitespace(cells=[Cell(x1=60, y1=25, x2=64, y2=35)])],
                             position=2)

    tb_area_3 = ImageSegment(x1=6, y1=100, x2=64, y2=110,
                             elements=[Cell(x1=10, y1=100, x2=20, y2=110),
                                       Cell(x1=30, y1=100, x2=40, y2=110),
                                       Cell(x1=50, y1=100, x2=60, y2=110)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=100, x2=10, y2=110)]),
                                          Whitespace(cells=[Cell(x1=20, y1=100, x2=30, y2=110)]),
                                          Whitespace(cells=[Cell(x1=40, y1=100, x2=50, y2=110)]),
                                          Whitespace(cells=[Cell(x1=60, y1=100, x2=64, y2=110)])],
                             position=4)

    assert coherent_table_areas(tb_area_1=tb_area_1, tb_area_2=tb_area_2, char_length=4, median_line_sep=10)
    assert not coherent_table_areas(tb_area_1=tb_area_1, tb_area_2=tb_area_3, char_length=4, median_line_sep=10)
    assert not coherent_table_areas(tb_area_1=tb_area_2, tb_area_2=tb_area_3, char_length=4, median_line_sep=10)


def test_table_segment_from_group():
    tb_group = [ImageSegment(x1=6, y1=10, x2=64, y2=20,
                             elements=[Cell(x1=10, y1=10, x2=20, y2=20),
                                       Cell(x1=30, y1=10, x2=40, y2=20),
                                       Cell(x1=50, y1=10, x2=60, y2=20)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=10, x2=10, y2=20)]),
                                          Whitespace(cells=[Cell(x1=20, y1=10, x2=30, y2=20)]),
                                          Whitespace(cells=[Cell(x1=40, y1=10, x2=50, y2=20)]),
                                          Whitespace(cells=[Cell(x1=60, y1=10, x2=64, y2=20)])],
                             position=1),
                ImageSegment(x1=6, y1=25, x2=64, y2=35,
                             elements=[Cell(x1=10, y1=25, x2=20, y2=35),
                                       Cell(x1=30, y1=25, x2=40, y2=35),
                                       Cell(x1=50, y1=25, x2=60, y2=35)],
                             whitespaces=[Whitespace(cells=[Cell(x1=6, y1=25, x2=10, y2=35)]),
                                          Whitespace(cells=[Cell(x1=20, y1=25, x2=30, y2=35)]),
                                          Whitespace(cells=[Cell(x1=40, y1=25, x2=50, y2=35)]),
                                          Whitespace(cells=[Cell(x1=60, y1=25, x2=64, y2=35)])],
                             position=2)]

    result = table_segment_from_group(table_segment_group=tb_group)

    expected = ImageSegment(x1=6, y1=10, x2=64, y2=35,
                            elements=[Cell(x1=10, y1=10, x2=20, y2=20), Cell(x1=30, y1=10, x2=40, y2=20),
                                      Cell(x1=50, y1=10, x2=60, y2=20), Cell(x1=10, y1=25, x2=20, y2=35),
                                      Cell(x1=30, y1=25, x2=40, y2=35), Cell(x1=50, y1=25, x2=60, y2=35)],
                            whitespaces=[Whitespace(cells=[Cell(x1=6, y1=10, x2=10, y2=20)]),
                                         Whitespace(cells=[Cell(x1=20, y1=10, x2=30, y2=20)]),
                                         Whitespace(cells=[Cell(x1=40, y1=10, x2=50, y2=20)]),
                                         Whitespace(cells=[Cell(x1=60, y1=10, x2=64, y2=20)]),
                                         Whitespace(cells=[Cell(x1=6, y1=25, x2=10, y2=35)]),
                                         Whitespace(cells=[Cell(x1=20, y1=25, x2=30, y2=35)]),
                                         Whitespace(cells=[Cell(x1=40, y1=25, x2=50, y2=35)]),
                                         Whitespace(cells=[Cell(x1=60, y1=25, x2=64, y2=35)])])

    assert result == expected


def test_get_table_segments():
    elements = [Cell(x1=10, y1=10, x2=20, y2=20), Cell(x1=30, y1=10, x2=40, y2=20), Cell(x1=50, y1=10, x2=60, y2=20),
                Cell(x1=10, y1=25, x2=20, y2=35), Cell(x1=30, y1=25, x2=40, y2=35), Cell(x1=50, y1=25, x2=60, y2=35),
                Cell(x1=10, y1=40, x2=20, y2=50), Cell(x1=50, y1=40, x2=60, y2=50),
                Cell(x1=10, y1=100, x2=20, y2=110), Cell(x1=30, y1=100, x2=40, y2=110),
                Cell(x1=50, y1=100, x2=60, y2=110),
                Cell(x1=10, y1=200, x2=20, y2=210), Cell(x1=30, y1=200, x2=40, y2=210),
                Cell(x1=50, y1=200, x2=60, y2=210)
                ]
    segment = ImageSegment(x1=0, y1=0, x2=1000, y2=1000, elements=elements)

    result = get_table_segments(segment=segment, char_length=4, median_line_sep=10)

    expected = [TableSegment(table_areas=[ImageSegment(x1=6, y1=10, x2=64, y2=20,
                                                       elements=[Cell(x1=10, y1=10, x2=20, y2=20),
                                                                 Cell(x1=30, y1=10, x2=40, y2=20),
                                                                 Cell(x1=50, y1=10, x2=60, y2=20)],
                                                       whitespaces=[Whitespace(cells=[Cell(x1=6, y1=10, x2=10, y2=20)]),
                                                                    Whitespace(cells=[Cell(x1=20, y1=10, x2=30, y2=20)]),
                                                                    Whitespace(cells=[Cell(x1=40, y1=10, x2=50, y2=20)]),
                                                                    Whitespace(cells=[Cell(x1=60, y1=10, x2=64, y2=20)])],
                                                       position=1),
                                          ImageSegment(x1=6, y1=25, x2=64, y2=35,
                                                       elements=[Cell(x1=10, y1=25, x2=20, y2=35),
                                                                 Cell(x1=30, y1=25, x2=40, y2=35),
                                                                 Cell(x1=50, y1=25, x2=60, y2=35)],
                                                       whitespaces=[Whitespace(cells=[Cell(x1=6, y1=25, x2=10, y2=35)]),
                                                                    Whitespace(cells=[Cell(x1=20, y1=25, x2=30, y2=35)]),
                                                                    Whitespace(cells=[Cell(x1=40, y1=25, x2=50, y2=35)]),
                                                                    Whitespace(cells=[Cell(x1=60, y1=25, x2=64, y2=35)])],
                                                       position=2),
                                          ImageSegment(x1=6, y1=40, x2=64, y2=50,
                                                       elements=[Cell(x1=10, y1=40, x2=20, y2=50),
                                                                 Cell(x1=50, y1=40, x2=60, y2=50)],
                                                       whitespaces=[Whitespace(cells=[Cell(x1=6, y1=40, x2=10, y2=50)]),
                                                                    Whitespace(cells=[Cell(x1=20, y1=40, x2=50, y2=50)]),
                                                                    Whitespace(cells=[Cell(x1=60, y1=40, x2=64, y2=50)])],
                                                       position=3)]),
                TableSegment(table_areas=[ImageSegment(x1=6, y1=100, x2=64, y2=110,
                                                       elements=[Cell(x1=10, y1=100, x2=20, y2=110),
                                                                 Cell(x1=30, y1=100, x2=40, y2=110),
                                                                 Cell(x1=50, y1=100, x2=60, y2=110)],
                                                       whitespaces=[Whitespace(cells=[Cell(x1=6, y1=100, x2=10, y2=110)]),
                                                                    Whitespace(cells=[Cell(x1=20, y1=100, x2=30, y2=110)]),
                                                                    Whitespace(cells=[Cell(x1=40, y1=100, x2=50, y2=110)]),
                                                                    Whitespace(cells=[Cell(x1=60, y1=100, x2=64, y2=110)])],
                                                       position=4)]),
                TableSegment(table_areas=[ImageSegment(x1=6, y1=200, x2=64, y2=210,
                                                       elements=[Cell(x1=10, y1=200, x2=20, y2=210),
                                                                 Cell(x1=30, y1=200, x2=40, y2=210),
                                                                 Cell(x1=50, y1=200, x2=60, y2=210)],
                                                       whitespaces=[Whitespace(cells=[Cell(x1=6, y1=200, x2=10, y2=210)]),
                                                                    Whitespace(cells=[Cell(x1=20, y1=200, x2=30, y2=210)]),
                                                                    Whitespace(cells=[Cell(x1=40, y1=200, x2=50, y2=210)]),
                                                                    Whitespace(cells=[Cell(x1=60, y1=200, x2=64, y2=210)])],
                                                       position=5)])
                ]

    assert result == expected
