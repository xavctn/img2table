# coding:utf-8
from typing import List, Optional

import cv2
import numpy as np
import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
from img2table.tables.processing.common import merge_contours, is_contained_cell


def compute_table_median_row_sep(table: Table, contours: List[Cell]) -> Optional[float]:
    """
    Compute median row separation in table
    :param table: Table object
    :param contours: list of image contours as cell objects
    :return: median row separation
    """
    # Create dataframe with contours
    list_elements = [{"id": idx, "x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2}
                     for idx, el in enumerate(contours)]
    df_elements = pl.LazyFrame(data=list_elements)

    # Filter on elements that are within the table
    df_elements_table = df_elements.filter((pl.col('x1') >= table.x1) & (pl.col('x2') <= table.x2)
                                           & (pl.col('y1') >= table.y1) & (pl.col('y2') <= table.y2))

    # Cross join to get corresponding elements and filter on elements that corresponds horizontally
    df_h_elms = (df_elements_table.join(df_elements_table, how='cross')
                 .filter(pl.col('id') != pl.col('id_right'))
                 .filter(pl.min_horizontal(['x2', 'x2_right']) - pl.max_horizontal(['x1', 'x1_right']) > 0)
                 )

    # Get element which is directly below
    df_elms_below = (df_h_elms.filter(pl.col('y1') < pl.col('y1_right'))
                     .sort(['id', 'y1_right'])
                     .with_columns(pl.lit(1).alias('ones'))
                     .with_columns(pl.col('ones').cum_sum().over(["id"]).alias('rk'))
                     .filter(pl.col('rk') == 1)
                     )

    if df_elms_below.collect().height == 0:
        return None

    # Compute median vertical distance between elements
    median_v_dist = (df_elms_below.with_columns(((pl.col('y1_right') + pl.col('y2_right')
                                                  - pl.col('y1') - pl.col('y2')) / 2).abs().alias('y_diff'))
                     .select(pl.median('y_diff'))
                     .collect()
                     .to_dicts()
                     .pop()
                     .get('y_diff')
                     )

    return median_v_dist


def handle_implicit_rows_table(img: np.ndarray, table: Table, contours: List[Cell], margin: int = 5) -> Table:
    """
    Find implicit rows and update tables based on those
    :param img: image array
    :param table: Table object
    :param contours: list of image contours as cell objects
    :param lines: list of lines in image
    :param margin: margin in pixels used for cropped images
    :return: reprocessed table with implicit rows
    """
    height, width = img.shape[:2]

    # If table is a single cell, do not search for implicit rows
    if table.nb_columns * table.nb_rows <= 1:
        return table

    # Get median row separation
    median_row_sep = compute_table_median_row_sep(table=table, contours=contours)

    if median_row_sep is None:
        return table

    list_splitted_rows = list()
    # Check if each row can be splitted
    for row in table.items:
        # If row is not vertically consistent, it is not relevant to split it
        if not row.v_consistent:
            list_splitted_rows.append(row)
            continue

        # Get cropped image
        cropped_img = img[max(row.y1 - margin, 0):min(row.y2 + margin, height),
                          max(row.x1 - margin, 0):min(row.x2 + margin, width)]

        # If cropped image is empty, do not do anything
        height_cropped, width_cropped = cropped_img.shape[:2]
        if height_cropped <= 0 or width_cropped <= 0:
            list_splitted_rows.append(row)
            continue

        # Reprocess images
        blur = cv2.GaussianBlur(cropped_img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

        # Dilate to combine adjacent text contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(median_row_sep // 3))))
        dilate = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Get list of contours
        list_cnts_cell = list()
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            x = x + row.x1 - margin
            y = y + row.y1 - margin
            contour_cell = Cell(x, y, x + w, y + h)
            list_cnts_cell.append(contour_cell)

        # Add contours to row
        row_cnts = merge_contours(contours=list_cnts_cell,
                                  vertically=True)

        # Delete contours that do not contains any elements
        filtered_contours = list()
        for row_cnt in row_cnts:
            # Get matching lines
            matching_els = [cnt for cnt in contours
                            if is_contained_cell(inner_cell=cnt, outer_cell=row_cnt, percentage=0.8)]

            if len(matching_els) == 0:
                continue
            filtered_contours.append(row_cnt)

        # Compute vertical delimiters
        vertical_delimiters = sorted([int(round((cnt_1.y2 + cnt_2.y1) / 2))
                                      for cnt_1, cnt_2 in zip(filtered_contours, filtered_contours[1:])])

        # Split row into multiple rows from vertical delimiters
        list_splitted_rows += row.split_in_rows(vertical_delimiters=vertical_delimiters)

    return Table(rows=list_splitted_rows)


def handle_implicit_rows(img: np.ndarray, tables: List[Table], contours: List[Cell]) -> List[Table]:
    """
    Detect and handle implicit rows in image tables
    :param img: image array
    :param tables: list of Table objects
    :param contours: list of image contours as cell objects
    :param lines: list of lines in image
    :return: list of Table objects updated taking into account implicit rows
    """
    # Detect implicit rows
    tables_implicit_rows = [handle_implicit_rows_table(img=img,
                                                       table=table,
                                                       contours=contours)
                            for table in tables]

    return tables_implicit_rows

