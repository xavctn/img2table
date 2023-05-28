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


def identify_table_lines(table: Table, lines: List[Line]) -> List[int]:
    """
    Identify horizontal rows that correspond to the table
    :param table: Table object
    :param lines: list of rows
    :return: list of y values corresponding to rows
    """
    # Get horizontal rows
    h_lines = [line for line in lines if line.horizontal]

    # Match rows with table rows
    y_lines = dict()
    y_values = sorted(list(set([row.y1 for row in table.items] + [row.y2 for row in table.items])))
    for line in h_lines:
        matching_y = sorted([y for y in y_values
                             if abs(line.y1 - y) / (table.height / table.nb_rows) <= 0.25],
                            key=lambda y: abs(line.y1 - y))
        if matching_y:
            y_val = matching_y.pop(0)
            y_lines[y_val] = y_lines.get(y_val, []) + [line]

    # Return y values where the line represent at least 75% of the table width
    final_lines = list()
    for k, v in y_lines.items():
        # Compute total length of rows
        x_vals = sorted(list(set([max(min(table.x2, l.x1), table.x1) for l in v]
                                 + [max(min(table.x2, l.x2), table.x1) for l in v])))
        total_length = sum([x_right - x_left for x_left, x_right in zip(x_vals, x_vals[1:])
                            if any([l for l in v if min(l.x2, x_right) - max(l.x1, x_left) > 0])])

        if total_length / table.width >= 0.75:
            final_lines.append(k)

    return final_lines


def check_header_coherency(header_rows: List[Row]) -> bool:
    """
    Check if detected header is coherent
    :param header_rows: list of rows creating the header
    :return: boolean indicating if the header is coherent
    """
    # Sort rows
    header_rows = sorted(header_rows, key=lambda r: r.y1, reverse=True)

    # Check if all columns of first header row are complete (except first one that can be missing)
    if not min([c.content for c in header_rows[0].items[1:]]):
        return False

    # Check coherent completeness of following rows
    complete_idx = set(list(range(header_rows[0].nb_columns)))
    for row in header_rows:
        complete_idx_row = {idx for idx, c in enumerate(row.items) if c.content}
        if len(complete_idx_row.difference(complete_idx)) > 0:
            return False
        complete_idx = complete_idx_row

    return True


def headers_from_lines(table: Table, lines: List[Line]) -> Table:
    """
    Detect potential headers from horizontal rows in image
    :param table: Table object
    :param lines: list of rows
    :return: table with processed header from rows
    """
    # Identify horizontal rows that correspond to the table
    y_values = identify_table_lines(table=table, lines=lines)

    # Create dict of rows indicating if all cells are complete (except the first one that can be missing in the header)
    dict_row_completeness = {(row.y1, row.y2): min([c.content for c in row.items]) for row in table.items}

    # Get rows that are in the top part of the table
    top_lines = [y for y in y_values if (y - table.y1) / table.height <= 0.25][:2]
    top_lines = sorted(list(set(top_lines if len(top_lines) == 2 else [table.y1] + top_lines)))

    if len(top_lines) == 2:
        # Check if there are no complete rows above the expected header
        if max([v for k, v in dict_row_completeness.items() if k[0] < top_lines[0]] + [False]):
            return table

        # Check if the header is coherent
        header_rows = [row for row in table.items
                       if top_lines[0] <= (row.y1 + row.y2) / 2 <= top_lines[1]]
        coherent_header = check_header_coherency(header_rows=header_rows)

        if coherent_header:
            # Create new header by replacing rows
            final_rows = [row for row in table.items if row.y1 >= top_lines[1]]
            new_row = Row(cells=[Cell(x1=c.x1, x2=c.x2, y1=top_lines[0], y2=top_lines[1])
                                 for c in table.items[0].items])
            final_rows.insert(0, new_row)
            return Table(rows=final_rows)

    return table


def process_headers(table: Table, lines: List[Line], elements: List[Cell]) -> Table:
    """
    Detect headers in table from rows and elements
    :param table: Table object
    :param lines: list of lines
    :param elements: list of elements
    :return: table with processed headers
    """
    # Check header coherency
    table = match_table_elements(table=table, elements=elements)

    # Get headers from line
    table_headers = headers_from_lines(table=table, lines=lines)

    return table_headers
