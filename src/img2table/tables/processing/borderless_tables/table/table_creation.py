# coding: utf-8
from typing import List, Optional

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.model import LineGroup


def reprocess_line_group(line_group: LineGroup, column_delimiters: List[Cell]) -> List[Cell]:
    """
    Get lines that correspond to column delimiters and merge overlapping lines
    :param line_group: group of lines as LineGroup object
    :param column_delimiters: list of column delimiters
    :return: list of lines
    """
    # Filter lines that correspond to column delimiters height
    y_min = min([delim.y1 for delim in column_delimiters])
    y_max = max([delim.y2 for delim in column_delimiters])
    filtered_lines = [line for line in line_group.lines if line.y1 >= y_min and line.y2 <= y_max]

    filtered_lines = sorted(filtered_lines, key=lambda l: l.v_center)

    # Merge overlapping lines
    seq = iter(filtered_lines)
    line = next(seq)
    new_lines = [Cell(x1=line.x1, y1=line.y1, x2=line.x2, y2=line.y2)]
    for line in seq:
        y_overlap = min(line.y2, new_lines[-1].y2) - max(line.y1, new_lines[-1].y1)
        if y_overlap / min(line.height, new_lines[-1].height) >= 0.25:
            new_lines[-1].x1 = min(new_lines[-1].x1, line.x1)
            new_lines[-1].y1 = min(new_lines[-1].y1, line.y1)
            new_lines[-1].x2 = max(new_lines[-1].x2, line.x2)
            new_lines[-1].y2 = max(new_lines[-1].y2, line.y2)
        else:
            new_lines.append(Cell(x1=line.x1, y1=line.y1, x2=line.x2, y2=line.y2))

    return new_lines


def get_table(lines: List[Cell], column_delimiters: List[Cell]) -> Optional[Table]:
    """
    Create table object from lines and column delimiters
    :param lines: list of lines
    :param column_delimiters: list of column delimiters
    :return: Table object if relevant
    """
    # Compute vertical delimiters
    lines = sorted(lines, key=lambda l: l.y1 + l.y2)
    y_min = min([line.y1 for line in lines])
    y_max = max([line.y2 for line in lines])
    v_delims = [y_min] + [round((up.y2 + down.y1) / 2) for up, down in zip(lines, lines[1:])] + [y_max]

    # Compute horizontal delimiters
    column_delimiters = sorted(column_delimiters, key=lambda l: l.x1 + l.x2)
    x_min = min([line.x1 for line in lines])
    x_max = max([line.x2 for line in lines])
    x_delims = [x_min] + [round((delim.x1 + delim.x2) / 2) for delim in column_delimiters] + [x_max]

    # Create table rows
    list_rows = list()
    for upper_bound, lower_bound in zip(v_delims, v_delims[1:]):
        l_cells = list()
        for l_bound, r_bound in zip(x_delims, x_delims[1:]):
            l_cells.append(Cell(x1=l_bound, y1=upper_bound, x2=r_bound, y2=lower_bound))
        list_rows.append(Row(cells=l_cells))

    # Create table
    table = Table(rows=list_rows)

    return table if table.nb_rows >= 2 and table.nb_columns >= 2 else None


def create_table(line_group: LineGroup, column_delimiters: List[Cell]) -> Optional[Table]:
    """
    Create table from lines and columns delimiters
    :param line_group: group of lines as LineGroup object
    :param column_delimiters: list of column delimiters
    :return: Table object if relevant
    """
    # If no delimiters are present, do not return anything
    if len(column_delimiters) == 0:
        return None

    # Reprocess lines
    reprocessed_lines = reprocess_line_group(line_group=line_group,
                                             column_delimiters=column_delimiters)

    # Create table object
    table = get_table(lines=reprocessed_lines, column_delimiters=column_delimiters)

    return table


