# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.cells import get_cells
from img2table.tables.processing.bordered_tables.tables import cluster_to_table
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow


def get_coherent_columns_dimensions(columns: DelimiterGroup, table_rows: List[TableRow]) -> DelimiterGroup:
    """
    Identify columns that encapsulate at least one row
    :param columns: column delimiters group
    :param table_rows: list of table rows
    :return: relevant columns according to table rows
    """
    original_delimiters = sorted(columns.delimiters, key=lambda delim: delim.x1 + delim.x2)

    # Get horizontal dimensions of rows
    x_min, x_max = min([row.x1 for row in table_rows]), max([row.x2 for row in table_rows])

    # Identify left and right delimiters
    left_delim = [delim for delim in original_delimiters if delim.x2 <= x_min][-1]
    right_delim = [delim for delim in original_delimiters if delim.x1 >= x_max][0]

    # Identify middle delimiters
    middle_delimiters = [delim for delim in original_delimiters if delim.x1 >= x_min and delim.x2 <= x_max]

    # Create new delimiter group
    delim_group = DelimiterGroup(delimiters=[left_delim] + middle_delimiters + [right_delim])

    return delim_group


def get_table(columns: DelimiterGroup, row_delimiters: List[Cell], contours: List[Cell]) -> Table:
    """
    Create table object from column delimiters and rows
    :param columns: column delimiters group
    :param row_delimiters: list of table row delimiters
    :param contours: list of image contours
    :return: Table object
    """
    # Convert delimiters to lines
    v_lines = [Line(x1=d.x1, x2=d.x2, y1=d.y1, y2=d.y2) for d in columns.delimiters]
    h_lines = [Line(x1=d.x1, x2=d.x2, y1=d.y1, y2=d.y2) for d in row_delimiters]

    # Identify cells
    cells = get_cells(horizontal_lines=h_lines, vertical_lines=v_lines)

    # Create table object
    table = cluster_to_table(cluster_cells=cells, elements=contours, borderless=True)

    return table if table.nb_columns >= 3 and table.nb_rows >= 2 else None
