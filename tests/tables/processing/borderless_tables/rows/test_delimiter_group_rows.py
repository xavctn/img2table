# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow
from img2table.tables.processing.borderless_tables.rows.delimiter_group_rows import \
    get_delimiter_group_row_separation, identify_rows, identify_delimiter_group_rows, \
    not_overlapping_rows, score_row_group, get_rows_from_overlapping_cluster


def test_get_delimiter_group_row_separation():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    result = get_delimiter_group_row_separation(delimiter_group=delimiter_group)

    assert result == 66


def test_not_overlapping_rows():
    tb_row_1 = TableRow(cells=[Cell(x1=0, x2=10, y1=0, y2=20)])
    tb_row_2 = TableRow(cells=[Cell(x1=0, x2=10, y1=8, y2=40)])
    tb_row_3 = TableRow(cells=[Cell(x1=0, x2=10, y1=36, y2=67)])

    assert not not_overlapping_rows(tb_row_1=tb_row_1, tb_row_2=tb_row_2)
    assert not_overlapping_rows(tb_row_1=tb_row_1, tb_row_2=tb_row_3)
    assert not not_overlapping_rows(tb_row_1=tb_row_2, tb_row_2=tb_row_3)


def test_score_row_group():
    row_group = [TableRow(cells=[Cell(x1=0, x2=10, y1=0, y2=20)]),
                 TableRow(cells=[Cell(x1=0, x2=10, y1=18, y2=43),
                                 Cell(x1=0, x2=10, y1=19, y2=45)])
                 ]

    result = score_row_group(row_group=row_group,
                             height=100,
                             max_elements=5)

    assert round(result, 2) == 0.27


def test_get_rows_from_overlapping_cluster():
    row_cluster = [TableRow(cells=[Cell(x1=0, x2=10, y1=0, y2=20)]),
                   TableRow(cells=[Cell(x1=20, x2=100, y1=0, y2=8)]),
                   TableRow(cells=[Cell(x1=20, x2=100, y1=11, y2=20), Cell(x1=20, x2=100, y1=11, y2=20)])]

    result = get_rows_from_overlapping_cluster(row_cluster=row_cluster)

    assert result == [TableRow(cells=[Cell(x1=20, x2=100, y1=0, y2=8)]),
                      TableRow(cells=[Cell(x1=20, x2=100, y1=11, y2=20), Cell(x1=20, x2=100, y1=11, y2=20)])]


def test_identify_rows():
    with open("test_data/delimiter_group.json", "r") as f:
        elements = [Cell(**c) for c in json.load(f).get('elements')]

    result = identify_rows(elements=elements,
                           ref_size=22)

    assert len(result) == 17
    assert min([row.y1 for row in result]) == 45
    assert max([row.y2 for row in result]) == 1147
    assert min([row.x1 for row in result]) == 93
    assert max([row.x2 for row in result]) == 1233


def test_identify_delimiter_group_rows():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    result, sep = identify_delimiter_group_rows(delimiter_group=delimiter_group)

    assert sep == 66
    assert len(result) == 17
    assert min([row.y1 for row in result]) == 45
    assert max([row.y2 for row in result]) == 1147
    assert min([row.x1 for row in result]) == 93
    assert max([row.x2 for row in result]) == 1233
