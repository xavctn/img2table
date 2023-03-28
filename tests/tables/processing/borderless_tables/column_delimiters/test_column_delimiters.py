# coding: utf-8
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.column_delimiters import identify_trivial_delimiters, \
    find_whitespaces, compute_vertical_delimiters, filter_coherent_dels, get_whitespace_column_delimiters
from img2table.tables.processing.borderless_tables.model import LineGroup, TableLine


def test_identify_trivial_delimiters():
    line_group = LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=100, y1=0, y2=100)])])
    elements = [Cell(x1=10, x2=20, y1=10, y2=20), Cell(x1=40, x2=50, y1=10, y2=20)]

    result = identify_trivial_delimiters(line_group=line_group,
                                         elements=elements)

    expected = [Cell(x1=20, x2=40, y1=0, y2=100)]

    assert result == expected


def test_find_whitespaces():
    line_group = LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=100, y1=0, y2=100)])])
    elements = [Cell(x1=10, x2=20, y1=10, y2=20), Cell(x1=40, x2=50, y1=10, y2=20)]

    result = find_whitespaces(line_group=line_group,
                              elements=elements)

    expected = [Cell(x1=0, y1=20, x2=100, y2=100),
                Cell(x1=0, y1=0, x2=100, y2=10),
                Cell(x1=50, y1=10, x2=100, y2=20),
                Cell(x1=20, y1=10, x2=40, y2=20),
                Cell(x1=0, y1=10, x2=10, y2=20)]

    assert result == expected


def test_compute_vertical_delimiters():
    whitespaces = [Cell(x1=0, x2=20, y1=0, y2=20),
                   Cell(x1=4, x2=13, y1=20, y2=40),
                   Cell(x1=20, x2=24, y1=0, y2=38),
                   Cell(x1=40, x2=60, y1=0, y2=31),
                   Cell(x1=60, x2=64, y1=0, y2=31)]

    result = compute_vertical_delimiters(whitespaces=whitespaces,
                                         ref_height=40)

    expected = [Cell(x1=4, y1=0, x2=13, y2=40),
                Cell(x1=20, y1=0, x2=24, y2=38),
                Cell(x1=40, y1=0, x2=64, y2=31)]

    assert result == expected


def test_filter_coherent_dels():
    v_delimiters = [Cell(x1=12, x2=23, y1=14, y2=57),
                    Cell(x1=35, x2=48, y1=17, y2=66),
                    Cell(x1=48, x2=61, y1=14, y2=46),
                    Cell(x1=61, x2=66, y1=17, y2=66),
                    Cell(x1=122, x2=143, y1=22, y2=66)]
    elements = [Cell(x1=0, x2=10, y1=0, y2=20), Cell(x1=110, x2=1200, y1=0, y2=20)]

    result = filter_coherent_dels(v_delimiters=v_delimiters,
                                  elements=elements)

    expected = [Cell(x1=12, y1=17, x2=23, y2=57),
                Cell(x1=35, y1=17, x2=48, y2=57),
                Cell(x1=61, y1=17, x2=66, y2=57)]

    assert result == expected


def test_get_whitespace_column_delimiters():
    line_group = LineGroup(lines=[TableLine(cells=[Cell(x1=0, x2=100, y1=0, y2=100)])])
    elements = [Cell(x1=10, x2=20, y1=10, y2=20), Cell(x1=40, x2=50, y1=10, y2=20)]

    result = get_whitespace_column_delimiters(line_group=line_group,
                                              segment_elements=elements)

    assert result == [Cell(x1=20, y1=0, x2=40, y2=100)]
