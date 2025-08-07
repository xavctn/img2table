
import numpy as np
import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.common import is_contained_cell


def normalize_table_cells(cluster_cells: list[Cell]) -> list[Cell]:
    """
    Normalize cells from table cells
    :param cluster_cells: list of cells that form a table
    :return: list of normalized cells
    """
    # Compute table shape
    width = max(map(lambda c: c.x2, cluster_cells)) - min(map(lambda c: c.x1, cluster_cells))
    height = max(map(lambda c: c.y2, cluster_cells)) - min(map(lambda c: c.y1, cluster_cells))

    # Get list of existing horizontal values
    h_values = sorted({x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]})
    # Compute delimiters by grouping close values together
    h_delims = [round(np.mean(h_group)) for h_group in
                np.split(h_values, np.where(np.diff(h_values) >= min(width * 0.02, 10))[0] + 1)]

    # Get list of existing vertical values
    v_values = sorted({y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]})
    # Compute delimiters by grouping close values together
    v_delims = [round(np.mean(v_group)) for v_group in
                np.split(v_values, np.where(np.diff(v_values) >= min(height * 0.02, 10))[0] + 1)]

    # Normalize all cells
    normalized_cells = []
    for cell in cluster_cells:
        normalized_cell = Cell(x1=sorted(h_delims, key=lambda d: abs(d - cell.x1)).pop(0),
                               x2=sorted(h_delims, key=lambda d: abs(d - cell.x2)).pop(0),
                               y1=sorted(v_delims, key=lambda d: abs(d - cell.y1)).pop(0),
                               y2=sorted(v_delims, key=lambda d: abs(d - cell.y2)).pop(0))
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

    # Identify elements corresponding to each cell
    df_elements = pl.DataFrame([{"x1_el": el.x1, "y1_el": el.y1, "x2_el": el.x2, "y2_el": el.y2, "area_el": el.area}
                                for el in elements])
    df_cells = (pl.DataFrame([{"id_row": id_row, "id_col": id_col, "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2}
                              for id_row, row in enumerate(table.items)
                              for id_col, c in enumerate(row.items)])
                .with_columns((pl.col("id_row").n_unique().over(["x1", "y1", "x2", "y2"]) > 1).alias("merged_col"),
                              (pl.col("id_col").n_unique().over(["x1", "y1", "x2", "y2"]) > 1).alias("merged_row"))
                )

    df_cells_elements = (
        df_cells.join(df_elements, how="cross")
        .with_columns((pl.min_horizontal(['x2', 'x2_el']) - pl.max_horizontal(['x1', 'x1_el'])).alias("x_overlap"),
                      (pl.min_horizontal(['y2', 'y2_el']) - pl.max_horizontal(['y1', 'y1_el'])).alias("y_overlap"))
        .with_columns(pl.max_horizontal(pl.col('x_overlap'), pl.lit(0)).alias('x_overlap'),
                      pl.max_horizontal(pl.col('y_overlap'), pl.lit(0)).alias('y_overlap'))
        .with_columns(((pl.col('x_overlap') * pl.col('y_overlap')) / pl.col('area_el') >= 0.6).alias('contains'))
        .group_by("id_row", "id_col", "merged_row", "merged_col")
        .agg(pl.col('contains').max())
    )

    # Identify empty rows and empty columns
    df_empty_rows = (df_cells_elements.group_by("id_row")
                     .agg(pl.col('contains').max(),
                          pl.when(~pl.col('merged_col')).then(pl.col('contains')).max().alias("single_contains"),
                          pl.col("merged_col").min())
                     )
    empty_rows = sorted([row.get("id_row") for row in df_empty_rows.to_dicts()
                         if not row.get("contains") or (not row.get('merged_col') and not row.get('single_contains'))])

    df_empty_cols = (df_cells_elements.group_by("id_col")
                     .agg(pl.col('contains').max(),
                          pl.when(~pl.col('merged_row')).then(pl.col('contains')).max().alias("single_contains"),
                          pl.col("merged_row").min())
                     )
    empty_cols = sorted([row.get("id_col") for row in df_empty_cols.to_dicts()
                         if not row.get("contains") or (not row.get('merged_row') and not row.get('single_contains'))])

    # Remove empty rows and empty columns
    table.remove_rows(row_ids=empty_rows)
    table.remove_columns(col_ids=empty_cols)

    return table


def cluster_to_table(cluster_cells: list[Cell], elements: list[Cell], borderless: bool = False) -> Table:
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
    for y_top, y_bottom in zip(v_delims, v_delims[1:]):
        # Get matching cell
        matching_cells = [c for c in cluster_cells
                          if min(c.y2, y_bottom) - max(c.y1, y_top) >= 0.9 * (y_bottom - y_top)]
        list_cells = []
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
            elif matching_cells:
                # Get x value of the closest matching cells
                x_value = sorted([x_val for cell in matching_cells for x_val in [cell.x1, cell.x2]],
                                 key=lambda x: min(abs(x - x_left), abs(x - x_right))).pop(0)
                list_cells.append(Cell(x1=x_value, y1=y_top, x2=x_value, y2=y_bottom))
            else:
                list_cells.append(default_cell)

        list_rows.append(Row(cells=list_cells))

    # Create table
    table = Table(rows=list_rows, borderless=borderless)

    # Remove empty/unnecessary rows and columns from the table, based on elements
    return remove_unwanted_elements(table=table, elements=elements)
