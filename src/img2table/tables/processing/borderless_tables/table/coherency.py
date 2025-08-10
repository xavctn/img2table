import numpy as np

from img2table.tables.objects.table import Table


def check_row_coherency(table: Table, median_line_sep: float) -> bool:
    """
    Check row coherency of table
    :param table: Table object
    :param median_line_sep: median line separation
    :return: boolean indicating if table row heights are coherent
    """
    if table.nb_rows < 2:
        return False

    # Get median row separation
    median_row_separation = np.median([(lower_row.y1 + lower_row.y2 - upper_row.y1 - upper_row.y2) / 2
                                       for upper_row, lower_row in zip(table.items, table.items[1:])])

    return median_row_separation >= median_line_sep / 3


def check_column_coherency(table: Table, char_length: float) -> bool:
    """
    Check column coherency of table
    :param table: Table object
    :param char_length: average character length
    :return: boolean indicating if table column widths are coherent
    """
    if table.nb_columns < 2:
        return False

    # Get column widths
    col_widths = []
    for idx in range(table.nb_columns):
        col_elements = [row.items[idx] for row in table.items]
        col_width = min([el.x2 for el in col_elements]) - max([el.x1 for el in col_elements])
        col_widths.append(col_width)

    return np.median(col_widths) >= 3 * char_length


def check_table_coherency(table: Table, median_line_sep: float, char_length: float) -> bool:
    """
    Check if table has coherent dimensions
    :param table: Table object
    :param median_line_sep: median line separation
    :param char_length: average character length
    :return: boolean indicating if table dimensions are coherent
    """
    # Check row coherency of table
    row_coherency = check_row_coherency(table=table,
                                        median_line_sep=median_line_sep)

    # Check column coherency of table
    column_coherency = check_column_coherency(table=table,
                                              char_length=char_length)

    return row_coherency and column_coherency
