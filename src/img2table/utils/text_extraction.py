# coding: utf-8
import copy
from typing import List

import numpy as np
import pytesseract

from img2table.objects.ocr import OCRPage
from img2table.objects.tables import Table, Line, Row
from img2table.utils.common import get_bounding_area_text


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

    # Loop over cell area
    for idx in range(len(tables)):
        # Retrieve Table object representing the "title area" above the cell area
        try:
            if idx == 0:
                upper_line = Line(line=(0, 0, width, 0))
                lower_line = Line(line=tables[idx].upper_bound)
            else:
                upper_line = Line(line=tables[idx - 1].lower_bound)
                lower_line = Line(line=tables[idx].upper_bound)
            title_tb = Table(rows=Row.from_horizontal_lines(line_1=upper_line, line_2=lower_line))
        except IndexError:
            tables[idx].set_title(title=None)
            continue

        # Get contours in title area
        tb_title_cnts = get_bounding_area_text(img=img,
                                               table=title_tb,
                                               margin=0,
                                               blur_size=5,
                                               kernel_size=9)

        if tb_title_cnts.items[0].contours:
            # Get lowest contour
            lowest_cnt = tb_title_cnts.items[0].contours[-1]

            # Get text from OCR
            title = ocr_page.get_text_cell(cell=lowest_cnt, margin=margin)
        else:
            title = None

        tables[idx].set_title(title=title)

    return tables


def get_text_tables(img: np.ndarray, ocr_page: OCRPage, tables: List[Table],
                    header_detection: bool = True) -> List[Table]:
    """
    Extract text from cell areas
    :param img: image array
    :param ocr_page: OCRPage object
    :param tables: list of Table objects
    :param header_detection: boolean indicating if header detection is performed
    :return: list of Table objects with parsed titles and dataframes
    """
    # Get title of tables
    title_tables = get_title_tables(img=img,
                                    tables=tables,
                                    ocr_page=ocr_page)

    # Get text corresponding to each cell of the cell areas
    data_tables = [table.get_text_ocr(ocr_page=ocr_page,
                                      img=copy.deepcopy(img),
                                      header_detection=header_detection)
                   for table in title_tables]

    # Filter on non empty tables
    non_empty_tables = [table for table in data_tables if table.data is not None]

    return non_empty_tables
