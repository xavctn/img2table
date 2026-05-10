from itertools import pairwise

import numpy as np

from img2table.tables.types import Cell, Row, Table


def remove_unwanted_elements(table: Table, elements: list[Cell]) -> Table:
    """
    Remove empty/unnecessary rows and columns from the table, based on elements
    :param table: input Table object
    :param elements: list of image elements
    :return: processed table
    """
    if len(elements) == 0 or table.nb_rows * table.nb_columns == 0:
        return Table(rows=[])

    # Create arrays
    cells = [
        (id_row, id_col, c.x1, c.y1, c.x2, c.y2)
        for id_row, row in enumerate(table.rows)
        for id_col, c in enumerate(row.cells)
    ]
    id_rows = np.array([cell[0] for cell in cells])
    id_cols = np.array([cell[1] for cell in cells])
    cell_coords = np.array([cell[2:] for cell in cells])
    elements_coords = np.array([[el.x1, el.y1, el.x2, el.y2] for el in elements])
    elements_area = np.array([el.area for el in elements])

    # Identify cells that are repeated across rows or columns, corresponding to merged cells
    unique_cells, inverse = np.unique(cell_coords, axis=0, return_inverse=True)
    merged_rows = np.zeros(cell_coords.shape[0], dtype=bool)
    merged_cols = np.zeros(cell_coords.shape[0], dtype=bool)
    for idx in range(unique_cells.shape[0]):
        mask = inverse == idx
        merged_rows[mask] = np.unique(id_cols[mask]).shape[0] > 1
        merged_cols[mask] = np.unique(id_rows[mask]).shape[0] > 1

    # Compute overlap between each cell and each element
    x_overlap = np.maximum(
        np.minimum(cell_coords[:, np.newaxis, 2], elements_coords[np.newaxis, :, 2])
        - np.maximum(cell_coords[:, np.newaxis, 0], elements_coords[np.newaxis, :, 0]),
        0,
    )
    y_overlap = np.maximum(
        np.minimum(cell_coords[:, np.newaxis, 3], elements_coords[np.newaxis, :, 3])
        - np.maximum(cell_coords[:, np.newaxis, 1], elements_coords[np.newaxis, :, 1]),
        0,
    )
    table_contains = (
        (elements_coords[:, 0] >= table.x1)
        & (elements_coords[:, 2] <= table.x2)
        & (elements_coords[:, 1] >= table.y1)
        & (elements_coords[:, 3] <= table.y2)
        & (elements_area > 0)
    )
    overlap_pct = np.divide(
        x_overlap * y_overlap,
        elements_area,
        out=np.zeros_like(x_overlap, dtype=float),
        where=elements_area > 0,
    )
    contains = (overlap_pct >= 0.6) & table_contains
    cell_contains = contains.any(axis=1)

    # Identify empty rows, ignoring content from cells merged across rows
    empty_rows = sorted(
        [
            id_row
            for id_row in np.unique(id_rows)
            if not cell_contains[id_rows == id_row].any()
            or (
                not merged_cols[id_rows == id_row].min()
                and not cell_contains[(id_rows == id_row) & ~merged_cols].any()
            )
        ]
    )

    # Identify empty columns, ignoring content from cells merged across columns
    empty_cols = sorted(
        [
            id_col
            for id_col in np.unique(id_cols)
            if not cell_contains[id_cols == id_col].any()
            or (
                not merged_rows[id_cols == id_col].min()
                and not cell_contains[(id_cols == id_col) & ~merged_rows].any()
            )
        ]
    )

    # Remove empty rows and empty columns
    table.remove_rows(row_ids=empty_rows)
    table.remove_columns(col_ids=empty_cols)

    return table


def cluster_to_table(cluster_cells: list[Cell], elements: list[Cell]) -> Table:
    """
    Convert a cell cluster to a Table object
    :param cluster_cells: list of cells that form a table
    :param elements: list of image elements
    :return: table with rows inferred from table cells
    """
    # Get list of vertical and horizontal delimiters
    v_delims = sorted({cell.y1 for cell in cluster_cells} | {cell.y2 for cell in cluster_cells})
    h_delims = sorted({cell.x1 for cell in cluster_cells} | {cell.x2 for cell in cluster_cells})

    # Create rows and cells
    list_rows = []
    for y_top, y_bottom in pairwise(v_delims):
        # Get matching cell
        matching_cells = [
            c
            for c in cluster_cells
            if min(c.y2, y_bottom) - max(c.y1, y_top) >= 0.9 * (y_bottom - y_top)
        ]
        list_cells = []
        for x_left, x_right in pairwise(h_delims):
            # Create default cell
            default_cell = Cell(x1=x_left, y1=y_top, x2=x_right, y2=y_bottom)

            # Check cells that contain the default cell
            containing_cell = min(
                [
                    c
                    for c in matching_cells
                    if (x_ovrlp := max(0, min(default_cell.x2, c.x2) - max(default_cell.x1, c.x1)))
                    > 0
                    and (y_ovrlp := max(0, min(default_cell.y2, c.y2) - max(default_cell.y1, c.y1)))
                    > 0
                    and x_ovrlp * y_ovrlp >= 0.9 * min(c.area, default_cell.area)
                ],
                key=lambda c: c.area,
                default=None,
            )

            # Append either a cell that contain the default cell
            if containing_cell is not None:
                list_cells.append(containing_cell)
            elif matching_cells:
                # Get x value of the closest matching cells
                x_value = min(
                    [x_val for cell in matching_cells for x_val in [cell.x1, cell.x2]],
                    key=lambda x: min(abs(x - x_left), abs(x - x_right)),
                )
                list_cells.append(Cell(x1=x_value, y1=y_top, x2=x_value, y2=y_bottom))
            else:
                list_cells.append(default_cell)

        list_rows.append(Row(cells=list_cells))

    # Create table
    table = Table(rows=list_rows)

    # Remove empty/unnecessary rows and columns from the table, based on elements
    return remove_unwanted_elements(table=table, elements=elements)
