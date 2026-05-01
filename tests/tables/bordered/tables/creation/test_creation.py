import json
from pathlib import Path

from img2table.tables.bordered.tables.creation.creation import (
    cluster_to_table,
    remove_unwanted_elements,
)
from img2table.tables.types import Cell, Row, Table


def test_remove_unwanted_elements() -> None:
    table = Table(
        rows=[
            Row(
                cells=[
                    Cell(x1=0, y1=0, x2=20, y2=20),
                    Cell(x1=20, y1=0, x2=40, y2=20),
                    Cell(x1=40, y1=0, x2=60, y2=20),
                ]
            ),
            Row(
                cells=[
                    Cell(x1=0, y1=20, x2=20, y2=40),
                    Cell(x1=20, y1=20, x2=40, y2=40),
                    Cell(x1=40, y1=20, x2=60, y2=40),
                ]
            ),
            Row(
                cells=[
                    Cell(x1=0, y1=40, x2=20, y2=60),
                    Cell(x1=20, y1=40, x2=40, y2=60),
                    Cell(x1=40, y1=40, x2=60, y2=60),
                ]
            ),
        ]
    )
    elements = [
        Cell(x1=25, y1=5, x2=35, y2=15),
        Cell(x1=45, y1=5, x2=55, y2=15),
        Cell(x1=25, y1=25, x2=35, y2=35),
        Cell(x1=45, y1=25, x2=55, y2=35),
    ]

    result = remove_unwanted_elements(table=table, elements=elements)

    expected = Table(
        rows=[
            Row(cells=[Cell(x1=20, y1=0, x2=40, y2=20), Cell(x1=40, y1=0, x2=60, y2=20)]),
            Row(cells=[Cell(x1=20, y1=20, x2=40, y2=40), Cell(x1=40, y1=20, x2=60, y2=40)]),
        ]
    )

    assert result == expected


def test_cluster_to_table() -> None:
    with Path("test_data/cell_clusters_normalized.json").open() as f:
        cell_clusters = [[Cell(**el) for el in cluster] for cluster in json.load(f)]
    with Path("test_data/contours.json").open() as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = [cluster_to_table(cluster, contours) for cluster in cell_clusters]

    with Path("test_data/tables_from_cells.json").open() as f:
        expected = [
            Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb]) for tb in json.load(f)
        ]

    assert result == expected
