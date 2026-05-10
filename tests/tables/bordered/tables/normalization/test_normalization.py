import json
from pathlib import Path

from img2table.tables.bordered.tables.normalization import normalize_table_cells
from img2table.tables.types import Cell


def test_normalize_table_cells() -> None:
    with Path("test_data/cells_clustered.json").open() as f:
        cell_clusters = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    result = [
        normalize_table_cells(cluster_cells=cell_cluster, char_length=5)
        for cell_cluster in cell_clusters
    ]

    with Path("test_data/cell_clusters_normalized.json").open() as f:
        expected = [[Cell(**el) for el in cluster] for cluster in json.load(f)]

    assert result == expected
