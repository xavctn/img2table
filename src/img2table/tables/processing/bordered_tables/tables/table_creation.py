# coding: utf-8
from typing import List

import numpy as np
import polars as pl

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
    h_delims = [int(round(np.mean(h_group))) for h_group in
                np.split(h_values, np.where(np.diff(h_values) >= min(width * 0.02, 10))[0] + 1)]

    # Get list of existing vertical values
    v_values = sorted(list(set([y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]])))
    # Compute delimiters by grouping close values together
    v_delims = [int(round(np.mean(v_group))) for v_group in
                np.split(v_values, np.where(np.diff(v_values) >= min(height * 0.02, 10))[0] + 1)]

    # Normalize all cells
    normalized_cells = list()
    for cell in cluster_cells:
        normalized_cell = Cell(x1=sorted(h_delims, key=lambda d: abs(d - cell.x1)).pop(0),
                               x2=sorted(h_delims, key=lambda d: abs(d - cell.x2)).pop(0),
                               y1=sorted(v_delims, key=lambda d: abs(d - cell.y1)).pop(0),
                               y2=sorted(v_delims, key=lambda d: abs(d - cell.y2)).pop(0))
        # Check if cell is not empty
        if cell.area > 0:
            normalized_cells.append(normalized_cell)

    return normalized_cells


def remove_unwanted_elements(table: Table, elements: List[Cell]) -> Table:
    """
    Remove empty/unnecessary rows and columns from the table, based on elements
    :param table: input Table object
    :param elements: list of image elements
    :return: processed table
    """
    # Identify elements corresponding to each cell
    df_elements = pl.LazyFrame([{"x1_el": el.x1, "y1_el": el.y1, "x2_el": el.x2, "y2_el": el.y2, "area_el": el.area}
                                for el in elements])
    df_cells = pl.LazyFrame([{"id_row": id_row, "id_col": id_col,  "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2}
                             for id_row, row in enumerate(table.items)
                             for id_col, c in enumerate(row.items)])
    df_cells_elements = (
        df_cells.join(df_elements, how="cross")
        .with_columns((pl.min_horizontal(['x2', 'x2_el']) - pl.max_horizontal(['x1', 'x1_el'])).alias("x_overlap"),
                      (pl.min_horizontal(['y2', 'y2_el']) - pl.max_horizontal(['y1', 'y1_el'])).alias("y_overlap"))
        .filter(pl.col('x_overlap') > 0,
                pl.col('y_overlap') > 0)
        .with_columns((pl.col('x_overlap') * pl.col('y_overlap')).alias('area_intersection'))
        .filter(pl.col('area_intersection') / pl.col('area_el') >= 0.6)
        .select("id_row", "id_col")
        .unique()
        .collect()
    )

    # Identify empty rows and empty columns
    empty_rows = [id_row for id_row in range(table.nb_rows)
                  if id_row not in [rec.get('id_row') for rec in df_cells_elements.to_dicts()]]
    empty_cols = [id_col for id_col in range(table.nb_columns)
                  if id_col not in [rec.get('id_col') for rec in df_cells_elements.to_dicts()]]

    # Remove empty rows and empty columns
    table.remove_rows(row_ids=empty_rows)
    table.remove_columns(col_ids=empty_cols)

    return table


def cluster_to_table(cluster_cells: List[Cell], elements: List[Cell], borderless: bool = False) -> Table:
    """
    Convert a cell cluster to a Table object
    :param cluster_cells: list of cells that form a table
    :param elements: list of image elements
    :param borderless: boolean indicating if the created table is borderless
    :return: table with rows inferred from table cells
    """
    # Get list of vertical delimiters
    v_delims = sorted(list(set([y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]])))

    # Get list of horizontal delimiters
    h_delims = sorted(list(set([x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]])))

    # Create rows and cells
    list_rows = list()
    for y_top, y_bottom in zip(v_delims, v_delims[1:]):
        # Get matching cell
        matching_cells = [c for c in cluster_cells
                          if min(c.y2, y_bottom) - max(c.y1, y_top) >= 0.9 * (y_bottom - y_top)]
        list_cells = list()
        for x_left, x_right in zip(h_delims, h_delims[1:]):
            # Create default cell
            default_cell = Cell(x1=x_left, y1=y_top, x2=x_right, y2=y_bottom)

            # Check cells that contain the default cell
            containing_cells = sorted([c for c in matching_cells
                                       if is_contained_cell(inner_cell=default_cell, outer_cell=c, percentage=0.9)],
                                      key=lambda c: c.area)

            # Append either a cell that contain the default cell
            if containing_cells:
                list_cells.append(containing_cells.pop(0))
            else:
                # Get x value of closest matching cells
                x_value = sorted([x_val for cell in matching_cells for x_val in [cell.x1, cell.x2]],
                                 key=lambda x: min(abs(x - x_left), abs(x - x_right))).pop(0)
                list_cells.append(Cell(x1=x_value, y1=y_top, x2=x_value, y2=y_bottom))

        list_rows.append(Row(cells=list_cells))

    # Create table
    table = Table(rows=list_rows, borderless=borderless)

    # Remove empty/unnecessary rows and columns from the table, based on elements
    processed_table = remove_unwanted_elements(table=table, elements=elements)

    return processed_table
