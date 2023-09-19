# coding: utf-8
from typing import List

from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow
from img2table.tables.processing.borderless_tables.rows.delimiter_group_rows import identify_delimiter_group_rows


def detect_delimiter_group_rows(delimiter_group: DelimiterGroup) -> List[TableRow]:
    """
    Identify list of rows corresponding to the delimiter group
    :param delimiter_group: column delimiters group
    :return: list of rows corresponding to the delimiter group
    """
    # Identify list of rows corresponding to the delimiter group
    table_rows, median_row_sep = identify_delimiter_group_rows(delimiter_group=delimiter_group)

    return table_rows
