# coding: utf-8
import statistics
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table


def handle_vertical_merged_cells(row: Row) -> List[Row]:
    """
    Handle vertically merged cells in row by creating multiple rows with duplicated cells
    :param row: Row object
    :return: list of rows taking into account merged cells
    """
    # Sort cells
    cells = sorted(row.items, key=lambda c: (c.x1, c.y1, c.x2))

    # Based on cells, group cells by columns / horizontal positions
    seq = iter(cells)
    cols = [[next(seq)]]
    for cell in seq:
        if abs(cols[-1][0].x1 - cell.x1) / row.width >= 0.02:
            cols.append([])
        cols[-1].append(cell)

    # Compute number of implicit rows in row and determine vertical delimiters
    nb_rows = max(map(len, cols))
    v_delimiters = [statistics.mean([(col[idx].y1 + col[idx].y2) / 2 for col in cols if len(col) == nb_rows])
                    for idx in range(nb_rows)]

    # Recompute each column by splitting/duplicating into rows
    recomputed_columns = list()
    for col in cols:
        if len(col) == nb_rows:
            recomputed_columns.append(col)
        else:
            _col = list()
            for delim in v_delimiters:
                intersecting_cells = [cell for cell in col if cell.y1 <= delim <= cell.y2]
                closest_cell = intersecting_cells.pop() if intersecting_cells else Cell(x1=col[0].x1, x2=col[0].x2,
                                                                                        y1=int(delim), y2=int(delim))
                _col.append(closest_cell)
            recomputed_columns.append(_col)

    # Create new rows
    new_rows = [Row(cells=[col[idx] for col in recomputed_columns])
                for idx in range(nb_rows)]

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
                intersecting_cells = [cell for cell in row.items if cell.x1 <= delim <= cell.x2]
                closest_cell = intersecting_cells.pop() if intersecting_cells else Cell(x1=int(delim),
                                                                                        x2=int(delim),
                                                                                        y1=row.items[0].y1,
                                                                                        y2=row.items[0].y1)
                _cells.append(closest_cell)
            new_rows.append(Row(cells=_cells))

    return Table(rows=new_rows)


def handle_merged_cells(table: Table) -> Table:
    """
    Handle merged cells in table
    :param table: Table object
    :return: Table object with merged cells handled
    """
    # Handle vertically merged cells
    table_v_merged = Table(rows=[_row for row in table.items
                                 for _row in handle_vertical_merged_cells(row=row)]
                           )

    # Handle horizontally merged cells
    table_h_merged = handle_horizontal_merged_cells(table=table_v_merged)

    return table_h_merged
