# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.tables.table_creation import normalize_table_cells, cluster_to_table, \
    remove_unwanted_elements


def test_normalize_table_cells():
    with open("test_data/cells_clustered.json", 'r') as f:
        cell_clusters = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    result = [normalize_table_cells(cluster_cells=cell_cluster) for cell_cluster in cell_clusters]

    with open("test_data/cell_clusters_normalized.json", "r") as f:
        expected = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    assert result == expected


def test_remove_unwanted_elements():
    table = Table(rows=[Row(cells=[Cell(x1=0, y1=0, x2=20, y2=20),
                                   Cell(x1=20, y1=0, x2=40, y2=20),
                                   Cell(x1=40, y1=0, x2=60, y2=20)]),
                        Row(cells=[Cell(x1=0, y1=20, x2=20, y2=40),
                                   Cell(x1=20, y1=20, x2=40, y2=40),
                                   Cell(x1=40, y1=20, x2=60, y2=40)]),
                        Row(cells=[Cell(x1=0, y1=40, x2=20, y2=60),
                                   Cell(x1=20, y1=40, x2=40, y2=60),
                                   Cell(x1=40, y1=40, x2=60, y2=60)])
                        ]
                  )
    elements = [Cell(x1=25, y1=5, x2=35, y2=15), Cell(x1=45, y1=5, x2=55, y2=15),
                Cell(x1=25, y1=25, x2=35, y2=35), Cell(x1=45, y1=25, x2=55, y2=35)]

    result = remove_unwanted_elements(table=table, elements=elements)

    expected = Table(rows=[Row(cells=[Cell(x1=20, y1=0, x2=40, y2=20),
                                      Cell(x1=40, y1=0, x2=60, y2=20)]),
                           Row(cells=[Cell(x1=20, y1=20, x2=40, y2=40),
                                      Cell(x1=40, y1=20, x2=60, y2=40)]),
                           ]
                     )

    assert result == expected


def test_cluster_to_table():
    with open("test_data/cell_clusters_normalized.json", "r") as f:
        cell_clusters = [[Cell(**el) for el in cluster] for cluster in json.load(f)]
    with open("test_data/contours.json", "r") as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = [cluster_to_table(cluster, contours) for cluster in cell_clusters]

    with open("test_data/tables_from_cells.json", "r") as f:
        expected = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb])
                    for tb in json.load(f)]

    assert result == expected
