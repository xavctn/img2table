# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
from img2table.tables.processing.tables.cell_clustering import cluster_cells_in_tables
from img2table.tables.processing.tables.table_creation import cluster_to_table, normalize_table_cells


def get_tables(cells: List[Cell]) -> List[Table]:
    """
    Identify and create Table object from list of image cells
    :param cells: list of cells found in image
    :return: list of Table objects inferred from cells
    """
    # Cluster cells into tables
    list_cluster_cells = cluster_cells_in_tables(cells=cells)

    # Normalize cells in clusters
    clusters_normalized = [normalize_table_cells(cluster_cells=cluster_cells)
                           for cluster_cells in list_cluster_cells]

    # Create tables from cells clusters
    tables = [cluster_to_table(cluster_cells=cluster_cells)
              for cluster_cells in clusters_normalized]

    return tables
