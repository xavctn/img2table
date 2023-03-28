# coding: utf-8
from typing import List, Optional

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.model import LineGroup
from img2table.tables.processing.borderless_tables.table.headers import process_headers
from img2table.tables.processing.borderless_tables.table.table_creation import create_table


def identify_table(line_group: LineGroup, column_delimiters: List[Cell], lines: List[Line],
                   elements: List[Cell]) -> Optional[Table]:
    """
    Identify potential tables from line group and column delimiters
    :param line_group: group of line as LineGroup object
    :param column_delimiters: list of column delimiters
    :param lines: list of lines in image
    :param elements: list of elements from image
    :return: Table object if relevant
    """
    # Create table from lines and columns delimiters
    table = create_table(line_group=line_group,
                         column_delimiters=column_delimiters)

    if table is None:
        return table

    # Identify headers
    table_headers = process_headers(table=table, lines=lines, elements=elements)

    # Reset table content
    for id_row, row in enumerate(table_headers.items):
        for id_col, col in enumerate(row.items):
            table_headers.items[id_row].items[id_col].content = None

    return table_headers
