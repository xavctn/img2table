# coding: utf-8
import statistics
from typing import List

from img2table.objects.tables import Table, Row, Cell


def adjacent_cells(cell_1: Cell, cell_2: Cell) -> bool:
    """
    Compute if two cells are adjacent
    :param cell_1: first cell object
    :param cell_2: second cell object
    :return: boolean indicating if cells are adjacent
    """
    # Check correspondence on vertical borders
    overlapping_y = max(0, min(cell_1.y2, cell_2.y2) - max(cell_1.y1, cell_2.y1))
    diff_x = min(abs(cell_1.x2 - cell_2.x1),
                 abs(cell_1.x1 - cell_2.x2),
                 abs(cell_1.x1 - cell_2.x1),
                 abs(cell_1.x2 - cell_2.x2))
    if overlapping_y > 5 and diff_x / max(cell_1.width, cell_2.width) <= 0.05:
        return True

    # Check correspondence on horizontal borders
    overlapping_x = max(0, min(cell_1.x2, cell_2.x2) - max(cell_1.x1, cell_2.x1))
    diff_y = min(abs(cell_1.y2 - cell_2.y1),
                 abs(cell_1.y1 - cell_2.y2),
                 abs(cell_1.y1 - cell_2.y1),
                 abs(cell_1.y2 - cell_2.y2))
    if overlapping_x > 5 and diff_y / max(cell_1.height, cell_2.height) <= 0.05:
        return True

    return False


def group_cells_in_tables(cells: List[Cell]) -> List[List[Cell]]:
    """
    Based on adjacent cells, create tables
    :param cells: list cells in image
    :return: list of list of cells, representing several group of cells that form a table
    """
    # Loop over all cells to create relationships between adjacent cells
    list_relations = list()
    for i in range(len(cells)):
        for j in range(i, len(cells)):
            adjacent = adjacent_cells(cells[i], cells[j])
            if adjacent:
                list_relations.append([i, j])

    # Create clusters of cells that corresponds to tables
    dict_clusters = dict()
    ii = 0
    for rel in sorted(list_relations):
        matching_clusters = [k for k, v in dict_clusters.items() if rel[0] in v or rel[1] in v]
        if len(matching_clusters) == 0:
            dict_clusters[str(ii)] = list(set(rel))
            ii += 1
        elif len(matching_clusters) == 1:
            key = matching_clusters[0]
            dict_clusters[key] = list(set(dict_clusters[key] + rel))
        else:
            new_val = rel + [el for k, v in dict_clusters.items() for el in v if k in matching_clusters]
            dict_clusters[str(ii)] = list(set(new_val))
            for key in matching_clusters:
                dict_clusters.pop(key, None)
            ii += 1

    # Create list of cells for each table
    list_table_cells = [[cells[idx] for idx in v] for k, v in dict_clusters.items()]

    return list_table_cells


def normalize_table_cells(table_cells: List[Cell]) -> List[Cell]:
    """
    Normalize cells from table cells
    :param table_cells: list of cells that form a table
    :return: list of normalized cells
    """
    # Compute table bounds
    l_bound_tb = min([cell.x1 for cell in table_cells])
    r_bound_tb = max([cell.x2 for cell in table_cells])
    up_bound_tb = min([cell.y1 for cell in table_cells])
    low_bound_tb = max([cell.y2 for cell in table_cells])

    normalized_cells = list()
    list_y_values = list()
    # For each cell, normalize its dimensions
    for cell in sorted(table_cells, key=lambda c: (c.y1, c.y2)):
        # If the cell is on the left border, set its left border as the table left border
        if (cell.x1 - l_bound_tb) / (r_bound_tb - l_bound_tb) <= 0.02:
            cell.x1 = l_bound_tb
        # If the cell is on the right border, set its right border as the table right border
        if (r_bound_tb - cell.x2) / (r_bound_tb - l_bound_tb) <= 0.02:
            cell.x2 = r_bound_tb

        # Check if upper bound of the table corresponds to an existing value.
        # If so, set the upper bound to the existing value
        close_upper_values = [y for y in list_y_values if abs(y - cell.y1) / (low_bound_tb - up_bound_tb) <= 0.02]
        if len(close_upper_values) == 1:
            cell.y1 = close_upper_values[0]
        else:
            list_y_values.append(cell.y1)

        # Check if lower bound of the table corresponds to an existing value.
        # If so, set the lower bound to the existing value
        close_lower_values = [y for y in list_y_values if abs(y - cell.y2) / (low_bound_tb - up_bound_tb) <= 0.02]
        if len(close_lower_values) == 1:
            cell.y2 = close_lower_values[0]
        else:
            list_y_values.append(cell.y2)

        normalized_cells.append(cell)

    return normalized_cells


def create_rows_table(table_cells: List[Cell]) -> Table:
    """
    Based on a list of cells, determine rows that compose the table and return table
    :param table_cells: list of cells that form a table
    :return: table with rows inferred from table cells
    """
    # Normalize cells
    normalized_cells = normalize_table_cells(table_cells=table_cells)

    # Sort cells
    sorted_cells = sorted(normalized_cells, key=lambda c: (c.y1, c.x1, c.y2))

    # Loop over cells and create rows based on corresponding vertical positions
    l_rows = list()
    for idx, cell in enumerate(sorted_cells):
        if idx == 0:
            row = Row(cells=cell)
        elif cell.y1 < row.y2:
            row = row.add_cells(cells=cell)
        else:
            if row not in l_rows:
                l_rows.append(row)
            row = Row(cells=cell)

    if row not in l_rows:
        l_rows.append(row)

    return Table(rows=l_rows)


def handle_vertical_merged_cells(row: Row) -> List[Row]:
    """
    Handle vertically merged cells in row by creating multiple rows with duplicated cells
    :param row: Row object
    :return: list of rows taking into account merged cells
    """
    # Sort cells
    cells = sorted(row.items, key=lambda c: (c.x1, c.y1, c.x2))

    # Based on cells, group cells by columns / horizontal positions
    cols = list()
    for idx, cell in enumerate(cells):
        if idx == 0:
            curr_col = [cell]
            curr_x = cell.x1
        elif abs(curr_x - cell.x1) / row.width >= 0.02:
            if curr_col not in cols:
                cols.append(curr_col)
            curr_col = [cell]
            curr_x = cell.x1
        else:
            curr_col.append(cell)
        if curr_col not in cols:
            cols.append(curr_col)

    # Compute number of implicit rows in row and determine vertical delimiters
    nb_rows = max([len(col) for col in cols])
    v_delimiters = [(cell.y1 + cell.y2) / 2 for cell in [col for col in cols if len(col) == nb_rows][0]]

    # Recompute each column by splitting/duplicating into rows
    new_cols = list()
    for col in cols:
        if len(col) == nb_rows:
            new_cols.append(col)
        else:
            _col = list()
            for delim in v_delimiters:
                intersecting_cells = [cell for cell in col
                                      if cell.y1 <= delim <= cell.y2]
                if intersecting_cells:
                    closest_cell = intersecting_cells[0]
                else:
                    closest_cell = Cell(x1=col[0].x1, x2=col[0].x1, y1=col[0].y1, y2=col[0].y1)
                _col.append(closest_cell)
            new_cols.append(_col)

    # Create new rows
    new_rows = [Row(cells=[col[idx] for col in new_cols]) for idx in range(nb_rows)]

    return new_rows


def handle_horizontal_merged_cells(table: Table) -> Table:
    """
    Handle horizontally merged cells in table by duplicating cells in affected rows
    :param table: Table object
    :return: Table object taking into account horizontally merged cells
    """
    # Compute number of columns and get delimiters
    nb_cols = max([row.nb_columns for row in table.items])
    list_delimiters = [[(cell.x1 + cell.x2) / 2 for cell in row.items] for row in table.items
                       if row.nb_columns == nb_cols]
    average_delimiters = [statistics.mean([delim[idx] for delim in list_delimiters]) for idx in range(nb_cols)]

    # Check if rows have the right number of columns, else duplicate cells
    new_rows = list()
    for row in table.items:
        if row.nb_columns == nb_cols:
            new_rows.append(row)
        else:
            _cells = list()
            for delim in average_delimiters:
                intersecting_cells = [cell for cell in row.items
                                      if cell.x1 <= delim <= cell.x2]
                if intersecting_cells:
                    closest_cell = intersecting_cells[0]
                else:
                    closest_cell = Cell(x1=row.items[0].x1, x2=row.items[0].x1, y1=row.items[0].y1, y2=row.items[0].y1)
                _cells.append(closest_cell)
            new_rows.append(Row(cells=_cells))

    return Table(rows=new_rows)


def create_table_from_table_cells(table_cells: List[Cell]) -> Table:
    """
    Create table from list of cells
    :param table_cells: list of cells that form a table
    :return: Table object
    """
    # Create table rows
    table_rows = create_rows_table(table_cells=table_cells)

    # Handle vertically merged cells
    table_v_merged = Table(rows=[_row for row in table_rows.items for _row in handle_vertical_merged_cells(row=row)])

    # Handle horizontally merged cells
    table_h_merged = handle_horizontal_merged_cells(table=table_v_merged)

    return table_h_merged


def get_tables(cells: List[Cell]) -> List[Table]:
    """
    Identify and create Table object from list of image cells
    :param cells: list of cells found in image
    :return: list of Table objects inferred from cells
    """
    # Group cells into tables
    list_table_cells = group_cells_in_tables(cells=cells)

    # Parse cells to table
    list_tables = [create_table_from_table_cells(table_cells=table_cells) for table_cells in list_table_cells]

    return list_tables
