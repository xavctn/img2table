# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
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


def get_table(columns: DelimiterGroup, table_rows: List[TableRow], contours: List[Cell]) -> Table:
    """
    Create table object from column delimiters and rows
    :param columns: column delimiters group
    :param table_rows: list of table rows
    :param contours: list of image contours
    :return: Table object
    """
    # Identify coherent column delimiters in relationship to table rows
    coherent_columns = get_coherent_columns_dimensions(columns=columns,
                                                       table_rows=table_rows)

    # Compute vertical delimiters from rows
    lines = sorted(table_rows, key=lambda l: l.v_center)
    y_min = min([line.y1 for line in lines])
    y_max = max([line.y2 for line in lines])
    v_delims = [y_min] + [int(round((up.y2 + down.y1) / 2)) for up, down in zip(lines, lines[1:])] + [y_max]

    # Create cells for table
    list_cells = list()
    for y_top, y_bottom in zip(v_delims, v_delims[1:]):
        # Identify delimiters that correspond vertically to rows
        line_delims = [d for d in coherent_columns.delimiters if
                       min(d.y2, y_bottom) - max(d.y1, y_top) > (y_bottom - y_top) // 2]

        # Sort line delimiters and compute horizontal delimiters
        line_delims = sorted(line_delims, key=lambda d: d.x1)
        h_delims = [line_delims[0].x2] + [(d.x1 + d.x2) // 2 for d in line_delims[1:-1]] + [line_delims[-1].x1]

        for x_left, x_right in zip(h_delims, h_delims[1:]):
            cell = Cell(x1=x_left,
                        y1=y_top,
                        x2=x_right,
                        y2=y_bottom)
            list_cells.append(cell)

    # Create table object
    table = cluster_to_table(cluster_cells=list_cells, elements=contours, borderless=True)

    return table if table.nb_columns >= 3 and table.nb_rows >= 2 else None
