# coding: utf-8
from typing import List, Optional

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.table_creation.coherency import check_coherency_rows, \
    check_versus_content
from img2table.tables.processing.borderless_tables.table_creation.columns import get_column_delimiters
from img2table.tables.processing.borderless_tables.table_creation.rows import get_row_delimiters


def create_table_from_clusters(tb_clusters: List[List[Cell]], segment_cells: List[Cell]) -> Optional[Table]:
    """
    Create table from aligned clusters forming it
    :param tb_clusters: list of aligned word clusters
    :param segment_cells: list of word cells comprised in image segment
    :return: Table object if relevant
    """
    # Identify column delimiters from clusters
    col_dels, col_clusters = get_column_delimiters(tb_clusters=tb_clusters)

    # Identify row delimiters from clusters
    row_dels = get_row_delimiters(tb_clusters=col_clusters)

    # Create rows
    list_rows = list()
    for upper_bound, lower_bound in zip(row_dels, row_dels[1:]):
        l_cells = list()
        for l_bound, r_bound in zip(col_dels, col_dels[1:]):
            l_cells.append(Cell(x1=l_bound, y1=upper_bound, x2=r_bound, y2=lower_bound))
        list_rows.append(Row(cells=l_cells))

    # Create table
    table = Table(rows=list_rows)

    # Check coherency of top and bottom rows of table which should have at least 2 columns with content
    coherent_table = check_coherency_rows(tb=table,
                                          word_cells=[cell for cl in col_clusters for cell in cl])

    # Check if table is mainly comprised of cells used for its creation
    final_table = check_versus_content(table=coherent_table,
                                       table_cells=[cell for cl in col_clusters for cell in cl],
                                       segment_cells=segment_cells)

    return final_table
