# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.bordered_tables.tables.cell_clustering import cluster_cells_in_tables


def test_cluster_cells_in_tables():
    with open("test_data/cells.json", 'r') as f:
        cells = [Cell(**el) for el in json.load(f)]

    result = cluster_cells_in_tables(cells=cells)

    with open("test_data/cells_clustered.json", 'r') as f:
        expected = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    assert all([cl in result for cl in expected])
    assert all([cl in expected for cl in result])
