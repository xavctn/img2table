# coding: utf-8
from typing import List

import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.common import is_contained_cell


def normalize_table_cells(cluster_cells: List[Cell]) -> List[Cell]:
    """
    Normalize cells from table cells
    :param cluster_cells: list of cells that form a table
    :return: list of normalized cells
    """
    # Compute table shape
    width = max(map(lambda c: c.x2, cluster_cells)) - min(map(lambda c: c.x1, cluster_cells))
    height = max(map(lambda c: c.y2, cluster_cells)) - min(map(lambda c: c.y1, cluster_cells))

    # Get list of existing horizontal values
    h_values = sorted(list(set([x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]])))
    # Compute delimiters by grouping close values together
    h_delims = [round(np.mean(h_group)) for h_group in
                np.split(h_values, np.where(np.diff(h_values) / height >= 0.02)[0] + 1)]

    # Get list of existing vertical values
    v_values = sorted(list(set([y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]])))
    # Compute delimiters by grouping close values together
    v_delims = [round(np.mean(v_group)) for v_group in
                np.split(v_values, np.where(np.diff(v_values) / width >= 0.02)[0] + 1)]

    # Normalize all cells
    normalized_cells = list()
    for cell in cluster_cells:
        normalized_cell = Cell(x1=sorted(h_delims, key=lambda d: abs(d - cell.x1)).pop(0),
                               x2=sorted(h_delims, key=lambda d: abs(d - cell.x2)).pop(0),
                               y1=sorted(v_delims, key=lambda d: abs(d - cell.y1)).pop(0),
                               y2=sorted(v_delims, key=lambda d: abs(d - cell.y2)).pop(0))
        # Check if cell is not empty
        if cell.height * cell.width > 0:
            normalized_cells.append(normalized_cell)

    return normalized_cells


def cluster_to_table(cluster_cells: List[Cell]) -> Table:
    """
    Convert a cell cluster to a Table object
    :param cluster_cells: list of cells that form a table
    :return: table with rows inferred from table cells
    """
    # Get list of vertical delimiters
    v_delims = sorted(list(set([y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]])))

    # Get list of horizontal delimiters
    h_delims = sorted(list(set([x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]])))

    # Create rows and cells
    list_rows = list()
    for y_top, y_bottom in zip(v_delims, v_delims[1:]):
        list_cells = list()
        for x_left, x_right in zip(h_delims, h_delims[1:]):
            # Create default cell
            default_cell = Cell(x1=x_left, y1=y_top, x2=x_right, y2=y_bottom)

            # Check cells that contain the default cell
            containing_cells = sorted([c for c in cluster_cells
                                       if is_contained_cell(inner_cell=default_cell, outer_cell=c, percentage=0.9)],
                                      key=lambda c: c.width * c.height)

            # Append either a cell that contain the default cell, or the default cell itself
            list_cells.append(containing_cells.pop(0) if containing_cells else default_cell)

        list_rows.append(Row(cells=list_cells))

    return Table(rows=list_rows)
