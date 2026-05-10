from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.tables.bordered.tables.creation.cell_clustering import (
    cluster_cells_in_tables,
)
from img2table.tables.bordered.tables.creation.creation import cluster_to_table
from img2table.tables.bordered.tables.misc.semi_bordered import (
    add_semi_bordered_cells,
)
from img2table.tables.bordered.tables.normalization import normalize_table_cells

if TYPE_CHECKING:
    from img2table.tables.types import Cell, Line, Table


def get_tables(
    cells: list[Cell], elements: list[Cell], lines: list[Line], char_length: float
) -> list[Table]:
    """
    Identify and create Table object from list of image cells
    :param cells: list of cells found in image
    :param elements: list of image elements
    :param lines: list of image lines
    :param char_length: average character length
    :return: list of Table objects inferred from cells
    """
    # Cluster cells into tables
    list_cluster_cells = cluster_cells_in_tables(cells=cells)

    # Normalize cells in clusters
    clusters_normalized: list[list[Cell]] = [
        normalize_table_cells(cluster_cells=cluster_cells, char_length=char_length)
        for cluster_cells in list_cluster_cells
    ]

    # Add semi-bordered cells to clusters
    complete_clusters: list[list[Cell]] = [
        add_semi_bordered_cells(cluster=cluster, lines=lines, char_length=char_length)
        for cluster in clusters_normalized
        if len(cluster) > 0
    ]

    # Create tables from cells clusters
    tables: list[Table] = [
        cluster_to_table(cluster_cells=cluster, elements=elements) for cluster in complete_clusters
    ]

    return [tb for tb in tables if tb.nb_rows * tb.nb_columns >= 2]
