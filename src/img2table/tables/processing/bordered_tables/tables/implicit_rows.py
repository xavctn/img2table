# coding:utf-8
import copy
from typing import List

import numpy as np

from img2table.tables.objects.table import Table
from img2table.tables.processing.common import get_contours_cell


def handle_implicit_rows_table(img: np.ndarray, table: Table) -> Table:
    """
    Find implicit rows and update tables based on those
    :param img: image array
    :param table: Table object
    :return: reprocessed table with implicit rows
    """
    # If table is a single cell, do not search for implicit rows
    if table.nb_columns * table.nb_rows <= 1:
        return table

    list_splitted_rows = list()
    # Check if each row can be splitted
    for row in table.items:
        # If row is not vertically consistent, it is not relevant to split it
        if not row.v_consistent:
            list_splitted_rows.append(row)
            continue

        # Compute contours
        contours = get_contours_cell(img=copy.deepcopy(img),
                                     cell=row,
                                     margin=-5,
                                     blur_size=5,
                                     kernel_size=5,
                                     merge_vertically=True)

        # Compute vertical delimiters
        vertical_delimiters = sorted([round((cnt_1.y2 + cnt_2.y1) / 2) for cnt_1, cnt_2 in zip(contours, contours[1:])])

        # Split row into multiple rows from vertical delimiters
        list_splitted_rows += row.split_in_rows(vertical_delimiters=vertical_delimiters)

    return Table(rows=list_splitted_rows)


def handle_implicit_rows(img: np.ndarray, tables: List[Table]) -> List[Table]:
    """
    Detect and handle implicit lines in image tables
    :param img: image array
    :param tables: list of Table objects
    :return: list of Table objects updated taking into account implicit rows
    """
    # Detect implicit rows
    tables_implicit_rows = [handle_implicit_rows_table(img=img, table=table) for table in tables]

    return tables_implicit_rows

