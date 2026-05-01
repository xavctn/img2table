import json
from pathlib import Path

from img2table.tables.bordered.tables.creation.cell_clustering import (
    cluster_cells_in_tables,
)
from img2table.tables.types import Cell


def test_cluster_cells_in_tables() -> None:
    with Path("test_data/cells.json").open() as f:
        cells = [Cell(**el) for el in json.load(f)]

    result = cluster_cells_in_tables(cells=cells)

    with Path("test_data/cells_clustered.json").open() as f:
        expected = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    result = map(set, result)
    expected = map(set, expected)

    assert all(cl in result for cl in expected)
    assert all(cl in expected for cl in result)
