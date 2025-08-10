from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.tables.cell_clustering import cluster_cells_in_tables
from img2table.tables.processing.bordered_tables.tables.semi_bordered import add_semi_bordered_cells
from img2table.tables.processing.bordered_tables.tables.table_creation import cluster_to_table, normalize_table_cells


def get_tables(cells: list[Cell], elements: list[Cell], lines: list[Line], char_length: float) -> list[Table]:
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
    clusters_normalized = [normalize_table_cells(cluster_cells=cluster_cells)
                           for cluster_cells in list_cluster_cells]

    # Add semi-bordered cells to clusters
    complete_clusters = [add_semi_bordered_cells(cluster=cluster, lines=lines, char_length=char_length)
                         for cluster in clusters_normalized if len(cluster) > 0]

    # Create tables from cells clusters
    tables = [cluster_to_table(cluster_cells=cluster, elements=elements)
              for cluster in complete_clusters]

    return [tb for tb in tables if tb.nb_rows * tb.nb_columns >= 2]
