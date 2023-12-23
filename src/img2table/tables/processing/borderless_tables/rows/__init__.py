# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup
from img2table.tables.processing.borderless_tables.rows.delimiter_group_rows import identify_delimiter_group_rows


def detect_delimiter_group_rows(delimiter_group: DelimiterGroup, contours: List[Cell]) -> List[Cell]:
    """
    Identify list of rows corresponding to the delimiter group
    :param delimiter_group: column delimiters group
    :param contours: list of image contours
    :return: list of rows corresponding to the delimiter group
    """
    # Identify list of rows corresponding to the delimiter group
    row_delimiters = identify_delimiter_group_rows(delimiter_group=delimiter_group,
                                                   contours=contours)

    return row_delimiters
