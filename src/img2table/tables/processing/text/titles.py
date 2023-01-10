# coding: utf-8
import copy
from typing import List

import numpy as np

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
from img2table.tables.processing.common import get_contours_cell


def get_title_tables(img: np.ndarray, tables: List[Table], ocr_df: OCRDataframe, margin: int = 5) -> List[Table]:
    """
    Retrieve titles of cell areas
    :param img: image array
    :param tables: list of Table objects
    :param ocr_df: OCRDataframe object
    :param margin: margin used
    :return: list of tables with title extracted
    """
    height, width = img.shape[:2]

    if len(tables) == 0:
        return []

    # Sort tables
    sorted_tables = sorted(tables, key=lambda tb: (tb.y1, tb.x1, tb.x2))

    # Cluster table vertically
    seq = iter(sorted_tables)
    tb_cl = [[next(seq)]]
    for tb in seq:
        if tb.y1 > tb_cl[-1][-1].y2:
            tb_cl.append([])
        tb_cl[-1].append(tb)

    # Identify relative zones for each title corresponding to each cluster
    final_tables = list()
    for id_cl, cluster in enumerate(tb_cl):
        # Compute horizontal boundaries of title
        x_delimiters = [10] + [round((tb_1.x2 + tb_2.x1) / 2) for tb_1, tb_2 in zip(cluster, cluster[1:])] + [width - 10]
        x_bounds = [(del_1, del_2) for del_1, del_2 in zip(x_delimiters, x_delimiters[1:])]

        # Compute vertical boundaries of title
        y_bounds = (max([tb.y2 for tb in tb_cl[id_cl - 1]]) if id_cl > 0 else 0, min([tb.y1 for tb in cluster]))

        # Fetch title for each table
        for id_tb, table in enumerate(cluster):
            # Get contours in title area
            cell_title = Cell(x1=x_bounds[id_tb][0], x2=x_bounds[id_tb][1], y1=y_bounds[0], y2=y_bounds[1])
            contours = get_contours_cell(img=copy.deepcopy(img),
                                         cell=cell_title,
                                         margin=0,
                                         blur_size=5,
                                         kernel_size=9)

            # Get text from OCR
            title = ocr_df.get_text_cell(cell=contours[-1], margin=margin) if contours else None

            table.set_title(title=title)
            final_tables.append(table)

    return final_tables
