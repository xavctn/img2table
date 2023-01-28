# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.tables.table_creation import normalize_table_cells, is_contained, cluster_to_table


def test_normalize_table_cells():
    with open("test_data/cells_clustered.json", 'r') as f:
        cell_clusters = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    result = [normalize_table_cells(cluster_cells=cell_cluster) for cell_cluster in cell_clusters]

    with open("test_data/cell_clusters_normalized.json", "r") as f:
        expected = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    assert result == expected


def test_is_contained():
    cell_1 = Cell(x1=100, y1=100, x2=200, y2=200)
    cell_2 = Cell(x1=0, y1=0, x2=300, y2=195)
    cell_3 = Cell(x1=1000, y1=1000, x2=2000, y2=2000)

    assert is_contained(inner_cell=cell_1, outer_cell=cell_2)
    assert not is_contained(inner_cell=cell_1, outer_cell=cell_3)


def test_cluster_to_table():
    with open("test_data/cell_clusters_normalized.json", "r") as f:
        cell_clusters = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    result = [cluster_to_table(cluster) for cluster in cell_clusters]

    with open("test_data/tables_from_cells.json", "r") as f:
        expected = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                    for tb in json.load(f)]

    assert result == expected
