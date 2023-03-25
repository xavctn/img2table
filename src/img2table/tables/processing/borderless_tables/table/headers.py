# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.common import is_contained_cell


def match_table_elements(table: Table, elements: List[Cell]) -> Table:
    """
    Identify for each table cell if some elements are present
    :param table: Table object
    :param elements: list of elements
    :return: table with content updated by elements
    """
    for id_row, row in enumerate(table.items):
        for id_col, cell in enumerate(row.items):
            table.items[id_row].items[id_col].content = max([is_contained_cell(inner_cell=el, outer_cell=cell)
                                                             for el in elements])

    return table


def check_header_coherency(table: Table, elements: List[Cell]) -> Table:
    """
    Check header coherency in order to restrict table height
    :param table: Table object
    :param elements: list of elements
    :return: Table with top rows coherent with an header
    """
    # Identify for each table cell if some elements are present
    table = match_table_elements(table=table, elements=elements)

    # Get first complete row
    first_complete_row = table.items[0]
    for row in table.items:
        if min([c.content for c in row.items]):
            first_complete_row = row
            break

    # Get all next rows in final rows
    final_rows = [row for row in table.items if row.y1 >= first_complete_row.y1]

    # Check if other rows are coherent
    prev_rows = [row for row in table.items if row.y1 < first_complete_row.y1]
    prev_rows = sorted(prev_rows, key=lambda r: r.y1, reverse=True)

    if prev_rows:
        complete_indices = set(range(table.nb_columns))
        for row in prev_rows:
            # Check if the row completeness is coherent with the previous ones
            row_indices = set([idx for idx, cell in enumerate(row.items) if cell.content])
            if len(row_indices.difference(complete_indices)) > 0:
                break
            final_rows.insert(0, row)
            complete_indices = row_indices

        # Create final table
        return Table(rows=final_rows)

    return table


def identify_table_lines(table: Table, lines: List[Line]) -> List[int]:
    """
    Identify horizontal lines that correspond to the table
    :param table: Table object
    :param lines: list of lines
    :return: list of y values corresponding to lines
    """
    # Get horizontal lines
    h_lines = [line for line in lines if line.horizontal]

    # Match lines with table rows
    y_lines = dict()
    y_values = sorted(list(set([row.y1 for row in table.items] + [row.y2 for row in table.items])))
    for line in h_lines:
        matching_y = [y for y in y_values
                      if abs(line.y1 - y) / (table.height / table.nb_rows) <= 0.25]
        if matching_y:
            y_val = matching_y.pop()
            y_lines[y_val] = y_lines.get(y_val, []) + [line]

    # Return y values where the line represent at least 75% of the table width
    final_lines = list()
    for k, v in y_lines.items():
        # Compute total length of lines
        x_vals = sorted(list(set([max(min(table.x2, l.x1), table.x1) for l in v]
                                 + [max(min(table.x2, l.x2), table.x1) for l in v])))
        total_length = sum([x_right - x_left for x_left, x_right in zip(x_vals, x_vals[1:])
                            if any([l for l in v if min(l.x2, x_right) - max(l.x1, x_left) > 0])])

        if total_length / table.width >= 0.75:
            final_lines.append(k)

    return final_lines


def headers_from_lines(table: Table, lines: List[Line]) -> Table:
    """
    Detect potential headers from horizontal lines in image
    :param table: Table object
    :param lines: list of lines
    :return: table with processed header from lines
    """
    # Identify horizontal lines that correspond to the table
    y_values = identify_table_lines(table=table, lines=lines)

    # Create dict of rows indicating if all cells are complete
    dict_row_completeness = {(row.y1, row.y2): min([c.content for c in row.items]) for row in table.items}

    # Get lines that are in the top part of the table
    top_lines = [y for y in y_values if (y - table.y1) / table.height <= 0.25][:2]
    top_lines = sorted(list(set(top_lines if len(top_lines) == 2 else [table.y1] + top_lines)))

    if len(top_lines) == 2:
        # Check if there are no complete lines above the expected header
        if max([v for k, v in dict_row_completeness.items() if k[0] < top_lines[0]] + [False]):
            return table

        # Check if the first line of the header is complete
        first_row_complete = [v for k, v in dict_row_completeness.items() if k[0] < top_lines[1]][-1]
        if first_row_complete:
            # Create new header by replacing lines
            final_rows = [row for row in table.items if row.y1 >= top_lines[1]]
            new_row = Row(cells=[Cell(x1=c.x1, x2=c.x2, y1=top_lines[0], y2=top_lines[1])
                                 for c in table.items[0].items])
            final_rows.insert(0, new_row)
            return Table(rows=final_rows)

    return table


def process_headers(table: Table, lines: List[Line], elements: List[Cell]) -> Table:
    """
    Detect headers in table from lines and elements
    :param table: Table object
    :param lines: list of lines
    :param elements: list of elements
    :return: table with processed headers
    """
    # Check header coherency
    table = check_header_coherency(table=table, elements=elements)

    # Get headers from line
    table_headers = headers_from_lines(table=table, lines=lines)

    return table_headers
