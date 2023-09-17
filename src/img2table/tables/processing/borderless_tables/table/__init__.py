# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow
from img2table.tables.processing.borderless_tables.table.table_creation import get_table


def identify_table(columns: DelimiterGroup, table_rows: List[TableRow], contours: List[Cell]) -> Table:
    """
    Identify table from column delimiters and rows
    :param columns: column delimiters group
    :param table_rows: list of table rows corresponding to columns
    :param contours: list of image contours
    :return: Table object
    """
    # Create table from rows and columns delimiters
    table = get_table(columns=columns,
                      table_rows=table_rows,
                      contours=contours)

    return table
