# coding: utf-8
from typing import List

from img2table.objects.tables import Line, Table, Row
from img2table.utils.common import intersection_bbox_line


def match_line_table(line: Line, table: Table, vertical_lines: List[Line]) -> Row:
    """
    Assess if line can be matched with a table
    :param line: line
    :param table: Table object
    :param vertical_lines: vertical lines
    :return: new Row object if line can be matched with table
    """
    # Check if left or right coordinates + length corresponds
    l_corresponds = abs((line.x1 - table.x1) / table.width) <= 0.02
    r_corresponds = abs((line.x2 - table.x2) / table.width) <= 0.02
    length_corresponds = 0.5 <= line.width / table.width <= 2
    if (l_corresponds or r_corresponds) and length_corresponds:
        # Create row with last line of table
        row = Row.from_horizontal_lines(line_1=line,
                                        line_2=Line(table.lower_bound))

        # If vertical lines intersect the row, add new row to table
        if vertical_lines:
            if max([intersection_bbox_line(row=row,
                                           line=line,
                                           without_border=False,
                                           horizontal_margin=5)
                    for line in vertical_lines]):
                return row
    return None


def update_table_with_line(line: Line, tables: List[Table], vertical_lines: List[Line]) -> List[Table]:
    """
    Match line with the most likely table and update list of tables
    :param line: line
    :param tables: list containing tables
    :param vertical_lines: vertical lines
    :return: updated tables list with new line
    """
    # Try to match line with :
    # - the lowest multi-line table
    # - single line tables that are under the lowest multi-line table
    multi_line_tables = sorted([(idx, table) for idx, table in enumerate(tables) if table.nb_rows > 0],
                               key=lambda t: t[1].y2, reverse=True)

    if multi_line_tables:
        multi_tb = multi_line_tables[0]
        single_line_tables = [(idx, table) for idx, table in enumerate(tables)
                              if table.nb_rows == 0 and table.y2 > multi_tb[1].y2]
        matching_tables = single_line_tables + [multi_tb]
    else:
        matching_tables = [(idx, table) for idx, table in enumerate(tables)]

    # Get eligible tables for lines
    eligible_tables = [(idx, match_line_table(line=line, table=table, vertical_lines=vertical_lines))
                       for idx, table in matching_tables]
    eligible_tables = [elig_table for elig_table in eligible_tables if elig_table[1] is not None]

    if len(eligible_tables) == 0:
        table = Table.from_horizontal_lines(line_1=line, line_2=line)
        tables += [table]
    else:
        best_table_idx, row = sorted(eligible_tables, key=lambda res: res[1].height)[0]
        tables[best_table_idx] = tables[best_table_idx].add_row(row)

    return tables


def get_tables(horizontal_lines: List[Line], vertical_lines: List[Line]) -> List[Table]:
    """
    Create tables based on horizontal lines
    :param horizontal_lines: horizontal lines
    :param vertical_lines: vertical lines
    :return: list of tables
    """
    # Set list of tables
    tables = list()

    # Loop over consecutive vertical lines to check if it is a table
    for idx, line in enumerate(horizontal_lines):
        # If first line, create new current table
        if idx == 0:
            tables = [Table.from_horizontal_lines(line, line)]
        else:
            tables = update_table_with_line(line, tables, vertical_lines)

    # Get tables with at least two rows
    list_tables = [table.normalize() for table in tables
                   if table.width * table.height > 0]

    return list_tables
