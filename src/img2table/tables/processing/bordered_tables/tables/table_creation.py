from collections import defaultdict
from itertools import pairwise

import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.common import _cluster_values


def is_contained_cell(
    inner_cell: Cell | tuple[int, int, int, int],
    outer_cell: Cell | tuple[int, int, int, int],
    percentage: float = 0.9,
) -> bool:
    """
    Assert if the inner cell is contained in outer cell
    :param inner_cell: inner cell
    :param outer_cell: Table object
    :param percentage: percentage of the inner cell that needs to be contained in the outer cell
    :return: boolean indicating if the inner cell is contained in the outer cell
    """
    # If needed, convert inner cell to Cell object
    inner_cell = Cell(*inner_cell) if not isinstance(inner_cell, Cell) else inner_cell
    # If needed, convert outer cell to Cell object
    outer_cell = Cell(*outer_cell) if not isinstance(outer_cell, Cell) else outer_cell

    # Compute common coordinates
    x_left = max(inner_cell.x1, outer_cell.x1)
    y_top = max(inner_cell.y1, outer_cell.y1)
    x_right = min(inner_cell.x2, outer_cell.x2)
    y_bottom = min(inner_cell.y2, outer_cell.y2)

    # Compute intersection area as well as inner cell area
    intersection_area = max(0, (x_right - x_left)) * max(0, (y_bottom - y_top))

    return intersection_area / inner_cell.area >= percentage


def normalize_table_cells(cluster_cells: list[Cell], char_length: float) -> list[Cell]:
    """
    Normalize cells from table cells
    :param cluster_cells: list of cells that form a table
    :param char_length: average character length
    :return: list of normalized cells
    """
    # Get list of existing horizontal values and cluster them
    h_values = sorted({x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]})
    cluster_mapping = defaultdict(list)
    for cl_idx, value in zip(
        _cluster_values(values=h_values, median_gap_multiple=0.1, min_gap=char_length),
        h_values,
        strict=True,
    ):
        cluster_mapping[cl_idx].append(value)
    # Get horizontal delimiters from cluster mapping
    h_delims = sorted(round(np.mean(vals)) for vals in cluster_mapping.values())

    # Get list of existing vertical values and cluster them
    v_values = sorted({y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]})
    cluster_mapping = defaultdict(list)
    for cl_idx, value in zip(
        _cluster_values(values=v_values, median_gap_multiple=0.1, min_gap=0.5 * char_length),
        v_values,
        strict=True,
    ):
        cluster_mapping[cl_idx].append(value)
    # Get vertical delimiters from cluster mapping
    v_delims = sorted(round(np.mean(vals)) for vals in cluster_mapping.values())

    # Normalize all cells
    normalized_cells: list[Cell] = []
    for cell in cluster_cells:
        normalized_cell = Cell(
            x1=min(h_delims, key=lambda d: abs(d - cell.x1)),
            x2=min(h_delims, key=lambda d: abs(d - cell.x2)),
            y1=min(v_delims, key=lambda d: abs(d - cell.y1)),
            y2=min(v_delims, key=lambda d: abs(d - cell.y2)),
        )
        # Check if cell is not empty
        if normalized_cell.area > 0:
            normalized_cells.append(normalized_cell)

    return normalized_cells


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
        for id_row, row in enumerate(table.items)
        for id_col, c in enumerate(row.items)
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


def cluster_to_table(
    cluster_cells: list[Cell], elements: list[Cell], borderless: bool = False
) -> Table:
    """
    Convert a cell cluster to a Table object
    :param cluster_cells: list of cells that form a table
    :param elements: list of image elements
    :param borderless: boolean indicating if the created table is borderless
    :return: table with rows inferred from table cells
    """
    # Get list of vertical delimiters
    v_delims = sorted({y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]})

    # Get list of horizontal delimiters
    h_delims = sorted({x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]})

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
            containing_cells = sorted(
                [
                    c
                    for c in matching_cells
                    if is_contained_cell(inner_cell=default_cell, outer_cell=c, percentage=0.9)
                ],
                key=lambda c: c.area,
            )

            # Append either a cell that contain the default cell
            if containing_cells:
                list_cells.append(containing_cells.pop(0))
            elif matching_cells:
                # Get x value of the closest matching cells
                x_value = sorted(
                    [x_val for cell in matching_cells for x_val in [cell.x1, cell.x2]],
                    key=lambda x: min(abs(x - x_left), abs(x - x_right)),
                ).pop(0)
                list_cells.append(Cell(x1=x_value, y1=y_top, x2=x_value, y2=y_bottom))
            else:
                list_cells.append(default_cell)

        list_rows.append(Row(cells=list_cells))

    # Create table
    table = Table(rows=list_rows, borderless=borderless)

    # Remove empty/unnecessary rows and columns from the table, based on elements
    return remove_unwanted_elements(table=table, elements=elements)
