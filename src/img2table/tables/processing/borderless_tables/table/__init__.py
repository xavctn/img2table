from typing import Optional

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.model import ColumnGroup
from img2table.tables.processing.borderless_tables.table.coherency import check_table_coherency
from img2table.tables.processing.borderless_tables.table.table_creation import get_table


def identify_table(columns: ColumnGroup, row_delimiters: list[Cell], contours: list[Cell], median_line_sep: float,
                   char_length: float) -> Optional[Table]:
    """
    Identify table from column delimiters and rows
    :param columns: column delimiters group
    :param row_delimiters: list of table row delimitres corresponding to columns
    :param contours: list of image contours
    :param median_line_sep: median line separation
    :param char_length: average character length
    :return: Table object
    """
    # Create table from rows and columns delimiters
    table = get_table(columns=columns,
                      row_delimiters=row_delimiters,
                      contours=contours)

    if table is not None and check_table_coherency(table=table,
                                                   median_line_sep=median_line_sep,
                                                   char_length=char_length):
            return table

    return None
