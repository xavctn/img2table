# coding: utf-8
import copy
from typing import List

import numpy as np

from img2table.objects.ocr import OCRPage
from img2table.objects.tables import Table, Cell
from img2table.utils.common import get_contours_cell


def get_title_tables(img: np.ndarray, tables: List[Table], ocr_page: OCRPage, margin: int = 5) -> List[Table]:
    """
    Retrieve titles of cell areas
    :param img: image array
    :param tables: list of Table objects
    :param ocr_page: OCRPage object
    :param margin: margin used
    :return: Title corresponding to the cell area
    """
    height, width, _ = img.shape
    
    if len(tables) == 0:
        return []
    
    # Sort tables
    sorted_tables = sorted(tables, key=lambda tb: (tb.y1, tb.x1, tb.x2))
    
    # Cluster table vertically
    tb_cl = list()
    for idx, tb in enumerate(sorted_tables):
        if idx == 0:
            cl = [tb]
        elif tb.y1 <= cl[-1].y2:
            cl.append(tb)
        else:
            tb_cl.append(cl)
            cl = [tb]
    tb_cl.append(cl)
    
    # Identify relative zones for each title corresponding to each cluster
    final_tables = list()
    for id_cl, cluster in enumerate(tb_cl):
        # Compute horizontal boundaries of title
        x_delimiters = [10] + [round((tb_1.x2 + tb_2.x1) / 2) for tb_1, tb_2 in zip(cluster, cluster[1:])] + [width - 10]
        x_bounds = [(del_1, del_2) for del_1, del_2 in zip(x_delimiters, x_delimiters[1:])]

        # Compute vertical boundaries of title
        if id_cl == 0:
            y_bounds = (0, min([tb.y1 for tb in cluster]))
        else:
            y_bounds = (max([tb.y2 for tb in tb_cl[id_cl - 1]]), min([tb.y1 for tb in cluster]))
        
        # Fetch title for each table
        for id_tb, table in enumerate(cluster):
            # Get contours in title area
            cell_title = Cell(x1=x_bounds[id_tb][0], x2=x_bounds[id_tb][1], y1=y_bounds[0], y2=y_bounds[1])
            contours = get_contours_cell(img=copy.deepcopy(img),
                                         cell=cell_title,
                                         margin=0,
                                         blur_size=5,
                                         kernel_size=9)
            
            if contours:
                # Get lowest contour
                lowest_cnt = contours[-1]

                # Get text from OCR
                title = ocr_page.get_text_cell(cell=lowest_cnt, margin=margin)
            else:
                title = None

            table.set_title(title=title)
            final_tables.append(table)

    return final_tables


def get_text_tables(img: np.ndarray, ocr_page: OCRPage, tables: List[Table]) -> List[Table]:
    """
    Extract text from cell areas
    :param img: image array
    :param ocr_page: OCRPage object
    :param tables: list of Table objects
    :return: list of Table objects with parsed titles and dataframes
    """
    # Get title of tables
    title_tables = get_title_tables(img=img,
                                    tables=tables,
                                    ocr_page=ocr_page)

    # Get text corresponding to each cell of the cell areas
    data_tables = [table.get_text_ocr(ocr_page=ocr_page)
                   for table in title_tables]

    # Filter on non empty tables
    non_empty_tables = [table for table in data_tables if table.data is not None]

    return non_empty_tables
