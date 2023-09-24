# coding: utf-8
from typing import List, Optional

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow
from img2table.tables.processing.borderless_tables.table.coherency import check_table_coherency
from img2table.tables.processing.borderless_tables.table.headers import process_headers
from img2table.tables.processing.borderless_tables.table.table_creation import get_table


def identify_table(columns: DelimiterGroup, table_rows: List[TableRow], lines: List[Line],
                   contours: List[Cell], median_line_sep: float, char_length: float) -> Optional[Table]:
    """
    Identify table from column delimiters and rows
    :param columns: column delimiters group
    :param table_rows: list of table rows corresponding to columns
    :param lines: list of lines in image
    :param contours: list of image contours
    :param median_line_sep: median line separation
    :param char_length: average character length
    :return: Table object
    """
    # Create table from rows and columns delimiters
    table = get_table(columns=columns,
                      table_rows=table_rows,
                      contours=contours)

    if table:
        # Process headers in table
        table_headers = process_headers(table=table,
                                        lines=lines,
                                        elements=contours)

        # Reset table content
        for id_row, row in enumerate(table_headers.items):
            for id_col, col in enumerate(row.items):
                table_headers.items[id_row].items[id_col].content = None

        if check_table_coherency(table=table_headers,
                                 median_line_sep=median_line_sep,
                                 char_length=char_length):
            return table_headers

    return None
