# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.table_creation import get_column_delimiters, get_row_delimiters, \
    check_versus_content, create_table_from_clusters


def test_get_column_delimiters():
    tb_clusters = [
        [Cell(x1=10, x2=20, y1=10, y2=20), Cell(x1=10, x2=20, y1=20, y2=30)],
        [Cell(x1=40, x2=60, y1=10, y2=20), Cell(x1=40, x2=60, y1=20, y2=30)],
        [Cell(x1=80, x2=100, y1=10, y2=20), Cell(x1=80, x2=100, y1=20, y2=30)],
    ]
    result = get_column_delimiters(tb_clusters=tb_clusters, margin=5)

    assert result == [5, 30, 70, 105]


def test_get_row_delimiters():
    tb_clusters = [
        [Cell(x1=10, x2=20, y1=10, y2=20), Cell(x1=10, x2=20, y1=20, y2=30)],
        [Cell(x1=40, x2=60, y1=10, y2=20), Cell(x1=40, x2=60, y1=20, y2=30)],
        [Cell(x1=80, x2=100, y1=10, y2=20), Cell(x1=80, x2=100, y1=20, y2=30)],
    ]
    result = get_row_delimiters(tb_clusters=tb_clusters, margin=5)

    assert result == [5, 20, 35]


def test_check_versus_content():
    table = Table(rows=Row(cells=[Cell(x1=0, y1=0, x2=100, y2=100)]))
    table_cells = [Cell(x1=0, y1=0, x2=100, y2=100), Cell(x1=0, y1=0, x2=100, y2=100)]
    segment_cells = [Cell(x1=0, y1=0, x2=100, y2=100), Cell(x1=0, y1=0, x2=100, y2=100)]

    assert check_versus_content(table=table, table_cells=table_cells, segment_cells=segment_cells) == table

    # Add some segment cells
    segment_cells += segment_cells
    assert check_versus_content(table=table, table_cells=table_cells, segment_cells=segment_cells) is None


def test_create_table_from_clusters():
    with open("test_data/clusters.json", "r") as f:
        clusters = [[Cell(**element) for element in cl] for cl in json.load(f)]

    with open("test_data/segment_cells.json", "r") as f:
        segment_cells = [Cell(**element) for element in json.load(f)]

    result = create_table_from_clusters(tb_clusters=clusters, segment_cells=segment_cells)

    expected = Table(rows=[Row(cells=[Cell(x1=35, y1=83, x2=218, y2=124),
                                      Cell(x1=218, y1=83, x2=447, y2=124),
                                      Cell(x1=447, y1=83, x2=621, y2=124)]),
                           Row(cells=[Cell(x1=35, y1=124, x2=218, y2=176),
                                      Cell(x1=218, y1=124, x2=447, y2=176),
                                      Cell(x1=447, y1=124, x2=621, y2=176)]),
                           Row(cells=[Cell(x1=35, y1=176, x2=218, y2=226),
                                      Cell(x1=218, y1=176, x2=447, y2=226),
                                      Cell(x1=447, y1=176, x2=621, y2=226)]),
                           Row(cells=[Cell(x1=35, y1=226, x2=218, y2=277),
                                      Cell(x1=218, y1=226, x2=447, y2=277),
                                      Cell(x1=447, y1=226, x2=621, y2=277)]),
                           Row(cells=[Cell(x1=35, y1=277, x2=218, y2=319),
                                      Cell(x1=218, y1=277, x2=447, y2=319),
                                      Cell(x1=447, y1=277, x2=621, y2=319)])
                           ])

    assert result == expected
