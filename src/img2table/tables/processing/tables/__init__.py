# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
from img2table.tables.processing.tables.cell_clustering import cluster_cells_in_tables
from img2table.tables.processing.tables.merged_cells import handle_merged_cells
from img2table.tables.processing.tables.rows_creation import create_rows_table


def get_tables(cells: List[Cell]) -> List[Table]:
    """
    Identify and create Table object from list of image cells
    :param cells: list of cells found in image
    :return: list of Table objects inferred from cells
    """
    # Cluster cells into tables
    list_cluster_cells = cluster_cells_in_tables(cells=cells)

    # Create tables from cells clusters
    tables = [create_rows_table(cluster_cells=cluster_cells)
              for cluster_cells in list_cluster_cells]

    # Handle merged cells in tables
    list_tables = [handle_merged_cells(table=table) for table in tables]

    return list_tables
