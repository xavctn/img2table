# coding:utf-8
import copy
from typing import List

import numpy as np

from img2table.objects.tables import Table
from img2table.utils.common import get_contours_cell


def handle_implicit_rows_table(white_img: np.ndarray, table: Table) -> Table:
    """
    Find implicit rows and update tables based on those
    :param white_img: white image array
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
        contours = get_contours_cell(img=copy.deepcopy(white_img),
                                     cell=row,
                                     margin=-5,
                                     blur_size=5,
                                     kernel_size=5,
                                     merge_vertically=True)

        # If row has no or one contour, it is not relevant to split it
        if len(contours) <= 1:
            list_splitted_rows.append(row)
            continue

        # Otherwise, compute vertical delimiters
        vertical_delimiters = [round((cnt_1.y2 + cnt_2.y1) / 2) for cnt_1, cnt_2 in zip(contours, contours[1:])]
        vertical_delimiters = sorted(vertical_delimiters)

        # Split row into multiple rows from vertical delimiters
        list_splitted_rows += row.split_in_rows(vertical_delimiters=vertical_delimiters)

    return Table(rows=list_splitted_rows)


def handle_implicit_rows(white_img: np.ndarray, tables: List[Table]) -> List[Table]:
    """
    Detect and handle implicit lines in image tables
    :param white_img: white image array
    :param tables: list of Table objects
    :return: list of Table objects updated taking into account implicit rows
    """
    tables_implicit_rows = [handle_implicit_rows_table(white_img=white_img, table=table)
                            for table in tables]

    return tables_implicit_rows
