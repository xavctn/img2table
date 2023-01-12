# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.tables.cell_clustering import cluster_cells_in_tables, adjacent_cells


def test_adjacent_cells():
    cell_1 = Cell(x1=0, x2=20, y1=0, y2=20)
    cell_2 = Cell(x1=21, x2=48, y1=10, y2=30)
    cell_3 = Cell(x1=14, x2=46, y1=19, y2=100)

    assert adjacent_cells(cell_1, cell_2)
    assert adjacent_cells(cell_1, cell_3)
    assert not adjacent_cells(cell_2, cell_3)


def test_cluster_cells_in_tables():
    with open("test_data/cells.json", 'r') as f:
        cells = [Cell(**el) for el in json.load(f)]

    result = cluster_cells_in_tables(cells=cells)

    with open("test_data/cells_clustered.json", 'r') as f:
        expected = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    assert result == expected
