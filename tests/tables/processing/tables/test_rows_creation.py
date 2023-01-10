# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.tables.rows_creation import normalize_table_cells, create_rows_table


def test_normalize_table_cells():
    with open("test_data/cells_clustered.json", 'r') as f:
        cell_clusters = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    result = [normalize_table_cells(cluster_cells=cell_cluster) for cell_cluster in cell_clusters]

    with open("test_data/cell_clusters_normalized.json", "r") as f:
        expected = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    assert result == expected


def test_create_rows_table():
    with open("test_data/cells_clustered.json", 'r') as f:
        cell_clusters = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    result = [create_rows_table(cluster_cells=cell_cluster) for cell_cluster in cell_clusters]

    with open("test_data/tables_from_cells.json", "r") as f:
        expected = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                    for tb in json.load(f)]

    assert result == expected
