# coding: utf-8
import copy
import statistics
from collections import Counter
from typing import List

from img2table.objects.tables import Cell, Row, Table


def columns_to_table(columns: List[List[Cell]], margin: int = 5) -> Table:
    """
    Create Table object from list of table columns
    :param columns: list of table columns
    :param margin: margin applied around outer cells composing the table
    :return: Table object
    """
    # From columns, compute the number of rows and columns
    nb_cols = len(columns)
    nb_rows = len(columns[0])

    # Compute row delimiters based on observed values
    row_delimiters = list()
    for idx in range(nb_rows - 1):
        min_value = max([col[idx].y2 for col in columns if col[idx] is not None])
        max_value = min([col[idx + 1].y1 for col in columns if col[idx + 1] is not None])
        row_delimiters.append(round((min_value + max_value) / 2))
    # Compute left and right ends and create final row delimiters
    left_end = min([col[0].y1 for col in columns if col[0] is not None]) - margin
    right_end = max([col[-1].y2 for col in columns if col[-1] is not None]) + margin
    row_delimiters = [left_end] + row_delimiters + [right_end]

    # Compute col delimiters based on observed values
    col_delimiters = list()
    for idx in range(nb_cols - 1):
        min_value = max([cell.x2 for cell in columns[idx] if cell is not None])
        max_value = min([cell.x1 for cell in columns[idx + 1] if cell is not None])
        col_delimiters.append(round((min_value + max_value) / 2))
    # Compute upper and lower ends and create final row delimiters
    upper_end = min([cell.x1 for cell in columns[0] if cell is not None]) - margin
    lower_end = max([cell.x2 for cell in columns[-1] if cell is not None]) + margin
    col_delimiters = [upper_end] + col_delimiters + [lower_end]

    # Compute ranges for each row and columns
    row_ranges = [el for el in zip(row_delimiters, row_delimiters[1:])]
    col_ranges = [el for el in zip(col_delimiters, col_delimiters[1:])]

    # Create cells and subsequently rows based on row and column ranges
    _rows = list()
    for row_rng in row_ranges:
        _cells = list()
        for col_rng in col_ranges:
            cell = Cell(x1=col_rng[0], y1=row_rng[0], x2=col_rng[1], y2=row_rng[1])
            _cells.append(cell)
        _rows.append(Row(cells=_cells))

    # Create table
    return Table(rows=_rows)


def cluster_group_to_table(cluster_group: List[List[Cell]]) -> Table:
    """
    Create Table object from a group of clusters representing a table
    :param cluster_group: group of cell clusters that each corresponds to a table
    :return: Table object
    """
    # Get most likely number of rows
    cnt_len_cluster = Counter([len(cluster) for cluster in cluster_group])
    nb_rows = sorted([k for k, v in cnt_len_cluster.items() if v == max(cnt_len_cluster.values())])[-1]

    # Determine vertical values
    vertical_values = [[(cluster[idx].y1 + cluster[idx].y2) / 2 for cluster in cluster_group if len(cluster) == nb_rows]
                       for idx in range(nb_rows)]
    vertical_values = [statistics.mean(val) for val in vertical_values]

    # For each cluster, assign a cell to a row in order to create a column
    columns = list()
    for cluster in sorted(cluster_group, key=lambda cl: statistics.mean([c.x1 + c.x2 for c in cl])):
        if len(cluster) == nb_rows:
            # If the cluster has the right number of rows, append it as a column
            columns.append(cluster)
        elif len(cluster) < nb_rows:
            # If the cluster has missing cells, determine for each cell the most likely row based on vertical position
            column = [None] * nb_rows
            for cell in cluster:
                most_likely_val = sorted(vertical_values,
                                         key=lambda v: abs((cell.y1 + cell.y2) / 2 - v))[0]
                column[vertical_values.index(most_likely_val)] = cell
            columns.append(column)
        else:
            # If the cluster has too many cells, for each row, determine the most likely cell based on vertical position
            column = [None] * nb_rows
            _cells = copy.deepcopy(cluster)
            for idx, value in enumerate(vertical_values):
                most_likely_cell = sorted(_cells,
                                          key=lambda x: abs((x.y1 + x.y2) / 2 - value))[0]
                column[idx] = most_likely_cell
                # Remove cell from list (a cell can not be used more than one time)
                _cells.remove(most_likely_cell)
            columns.append(column)

    # Create table from columns
    return columns_to_table(columns=columns)
