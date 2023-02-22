# coding: utf-8

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.table_creation.rows import get_row_delimiters, \
    merge_aligned_in_column, cluster_cells_centers, CellsCenter, unify_clusters, row_delimiters_from_borders


def test_merge_aligned_in_column():
    tb_clusters = [[Cell(x1=11, x2=23, y1=12, y2=93), Cell(x1=66, x2=73, y1=8, y2=87)],
                   [Cell(x1=100, x2=200, y1=0, y2=100)]]

    result = merge_aligned_in_column(tb_clusters=tb_clusters)

    expected = [[Cell(x1=11, x2=73, y1=8, y2=93)],
                [Cell(x1=100, x2=200, y1=0, y2=100)]]
    assert result == expected


def test_cluster_cells_centers():
    cells_centers = [CellsCenter(cells=[Cell(x1=0, x2=0, y1=0, y2=10)],
                                 cluster_id=0),
                     CellsCenter(cells=[Cell(x1=0, x2=0, y1=0, y2=10)],
                                 cluster_id=1),
                     CellsCenter(cells=[Cell(x1=0, x2=0, y1=10, y2=20), Cell(x1=0, x2=0, y1=12, y2=23)],
                                 cluster_id=0),
                     CellsCenter(cells=[Cell(x1=0, x2=0, y1=13, y2=17)],
                                 cluster_id=1),
                     CellsCenter(cells=[Cell(x1=0, x2=0, y1=53, y2=68)],
                                 cluster_id=0),
                     CellsCenter(cells=[Cell(x1=0, x2=0, y1=58, y2=68)],
                                 cluster_id=0)
                     ]

    result = cluster_cells_centers(cells_centers=cells_centers)

    expected = [
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=0, y2=10)],
                     cluster_id=0),
         CellsCenter(cells=[Cell(x1=0, x2=0, y1=0, y2=10)],
                     cluster_id=1)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=13, y2=17)],
                     cluster_id=1),
         CellsCenter(cells=[Cell(x1=0, x2=0, y1=10, y2=20), Cell(x1=0, x2=0, y1=12, y2=23)],
                     cluster_id=0)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=53, y2=68)],
                     cluster_id=0)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=58, y2=68)],
                     cluster_id=0)]
    ]

    assert result == expected


def test_unify_clusters():
    cl_unique_c_center = [
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=0, y2=10)],
                     cluster_id=0),
         CellsCenter(cells=[Cell(x1=0, x2=0, y1=1, y2=10)],
                     cluster_id=1)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=13, y2=17)],
                     cluster_id=1)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=53, y2=68)],
                     cluster_id=0)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=58, y2=68)],
                     cluster_id=0)]
    ]

    cl_merged_c_center = [
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=0, y2=10), Cell(x1=0, x2=0, y1=1, y2=10)],
                     cluster_id=0)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=0, y2=10), Cell(x1=0, x2=0, y1=53, y2=68)],
                     cluster_id=0)],
    ]

    result = unify_clusters(cl_unique_c_center=cl_unique_c_center,
                            cl_merged_c_center=cl_merged_c_center)

    expected = [
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=0, y2=10)],
                     cluster_id=0),
         CellsCenter(cells=[Cell(x1=0, x2=0, y1=1, y2=10)],
                     cluster_id=1)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=13, y2=17)],
                     cluster_id=1)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=53, y2=68)],
                     cluster_id=0)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=58, y2=68)],
                     cluster_id=0)],
        [CellsCenter(cells=[Cell(x1=0, x2=0, y1=0, y2=10), Cell(x1=0, x2=0, y1=1, y2=10)],
                     cluster_id=0)]
    ]

    assert result == expected


def test_row_delimiters_from_borders():
    borders = [(0, 10), (20, 30), (40, 50), (42, 56)]

    result = row_delimiters_from_borders(borders=borders, margin=5)

    assert result == [-5, 15, 35, 61]


def test_get_row_delimiters():
    tb_clusters = [
        [Cell(x1=10, x2=20, y1=10, y2=20), Cell(x1=10, x2=20, y1=20, y2=30)],
        [Cell(x1=40, x2=60, y1=10, y2=20), Cell(x1=40, x2=60, y1=20, y2=30)],
        [Cell(x1=80, x2=100, y1=10, y2=20), Cell(x1=80, x2=100, y1=20, y2=30)],
    ]
    result = get_row_delimiters(tb_clusters=tb_clusters, margin=5)

    assert result == [5, 20, 35]
