# coding: utf-8
from typing import List

from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow
from img2table.tables.processing.borderless_tables.table.headers import process_headers
from img2table.tables.processing.borderless_tables.table.table_creation import get_table


def identify_table(columns: DelimiterGroup, table_rows: List[TableRow], lines: List[Line]) -> Table:
    """
    Identify table from column delimiters and rows
    :param columns: column delimiters group
    :param table_rows: list of table rows corresponding to columns
    :param lines: list of detected solid rows in image
    :return: Table object
    """
    # Create table from rows and columns delimiters
    table = get_table(columns=columns,
                      table_rows=table_rows)

    # Identify headers
    table_headers = process_headers(table=table,
                                    lines=lines,
                                    elements=columns.elements)

    # Reset table content
    for id_row, row in enumerate(table_headers.items):
        for id_col, col in enumerate(row.items):
            table_headers.items[id_row].items[id_col].content = None

    return table_headers
