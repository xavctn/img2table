# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table


def normalize_table_cells(cluster_cells: List[Cell]) -> List[Cell]:
    """
    Normalize cells from table cells
    :param cluster_cells: list of cells that form a table
    :return: list of normalized cells
    """
    # Compute table bounds
    l_bound_tb = min(map(lambda c: c.x1, cluster_cells))
    r_bound_tb = max(map(lambda c: c.x2, cluster_cells))
    up_bound_tb = min(map(lambda c: c.y1, cluster_cells))
    low_bound_tb = max(map(lambda c: c.y2, cluster_cells))

    normalized_cells = list()
    list_y_values = list()
    # For each cell, normalize its dimensions
    for cell in sorted(cluster_cells, key=lambda c: (c.y1, c.y2)):
        # If the cell is on the left border, set its left border as the table left border
        if (cell.x1 - l_bound_tb) / (r_bound_tb - l_bound_tb) <= 0.02:
            cell.x1 = l_bound_tb
        # If the cell is on the right border, set its right border as the table right border
        if (r_bound_tb - cell.x2) / (r_bound_tb - l_bound_tb) <= 0.02:
            cell.x2 = r_bound_tb

        # Check if upper bound of the table corresponds to an existing value.
        # If so, set the upper bound to the existing value
        close_upper_values = [y for y in list_y_values
                              if abs(y - cell.y1) / (low_bound_tb - up_bound_tb) <= 0.02]
        if len(close_upper_values) == 1:
            cell.y1 = close_upper_values.pop()
        else:
            list_y_values.append(cell.y1)

        # Check if lower bound of the table corresponds to an existing value.
        # If so, set the lower bound to the existing value
        close_lower_values = [y for y in list_y_values
                              if abs(y - cell.y2) / (low_bound_tb - up_bound_tb) <= 0.02]
        if len(close_lower_values) == 1:
            cell.y2 = close_lower_values.pop()
        else:
            list_y_values.append(cell.y2)

        normalized_cells.append(cell)

    return normalized_cells


def create_rows_table(cluster_cells: List[Cell]) -> Table:
    """
    Based on a list of cells, determine rows that compose the table and return table
    :param cluster_cells: list of cells that form a table
    :return: table with rows inferred from table cells
    """
    # Normalize cells
    normalized_cells = normalize_table_cells(cluster_cells=cluster_cells)

    # Sort cells
    sorted_cells = sorted(normalized_cells, key=lambda c: (c.y1, c.x1, c.y2))

    # Loop over cells and create rows based on corresponding vertical positions
    seq = iter(sorted_cells)
    list_rows = [Row(cells=next(seq))]
    for cell in seq:
        if cell.y1 < list_rows[-1].y2:
            list_rows[-1].add_cells(cells=cell)
        else:
            list_rows.append(Row(cells=cell))

    return Table(rows=list_rows)

