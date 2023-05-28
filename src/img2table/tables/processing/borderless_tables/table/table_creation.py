# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.tables import cluster_to_table
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow


def get_table(columns: DelimiterGroup, table_rows: List[TableRow]) -> Table:
    """
    Create table object from column delimiters and rows
    :param columns: column delimiters group
    :param table_rows: list of table rows
    :return: Table object
    """
    # Compute vertical delimiters from rows
    lines = sorted(table_rows, key=lambda l: l.v_center)
    y_min = min([line.y1 for line in lines])
    y_max = max([line.y2 for line in lines])
    v_delims = [y_min] + [round((up.y2 + down.y1) / 2) for up, down in zip(lines, lines[1:])] + [y_max]

    # Create cells for table
    list_cells = list()
    for y_top, y_bottom in zip(v_delims, v_delims[1:]):
        # Identify delimiters that correspond vertically to rows
        line_delims = [d for d in columns.delimiters if
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
    table = cluster_to_table(list_cells)

    return table
