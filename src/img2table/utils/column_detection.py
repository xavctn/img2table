# coding:utf-8
import statistics
from typing import List

import numpy as np

from img2table.objects.tables import Table, Line, Cell
from img2table.utils.common import intersection_bbox_line, get_bounding_area_text


def get_columns_vertical_lines(table: Table, vertical_lines: List[Line]) -> List[int]:
    """
    Identify horizontal position of columns based on vertical lines
    :param table: list of horizontal lines constituting a table
    :param vertical_lines: vertical lines
    :return: list of horizontal position of columns
    """
    # Get vertical lines that are within the table bounding box
    table_v_lines = [line for line in vertical_lines
                     if table.x1 <= line.x1 <= table.x2
                     and (table.y1 <= line.y1 <= table.y2
                          or table.y1 <= line.y2 <= table.y2)]
    # Merge vertically aligned lines
    table_v_lines = sorted(table_v_lines, key=lambda l: (l.x1, l.y1, l.y2))
    merged_lines = list()

    if table_v_lines:
        for idx, line in enumerate(table_v_lines):
            if idx == 0:
                curr_line = line
            elif line.x1 == curr_line.x1:
                curr_line.y2 = curr_line.y2 + line.height
            else:
                merged_lines.append(curr_line)
                curr_line = line
        merged_lines.append(curr_line)

    # Get crossing lines
    crossing_lines = [line for line in merged_lines
                      if intersection_bbox_line(Cell(*table.bbox()), line)]

    return sorted(list(set([line.x1 for line in crossing_lines])))


def get_delimiter_min_width(table: Table, delimiters: List[int], min_width_column: int = 15) -> List[int]:
    """
    Filter column delimiters values to ensure that columns have a set minimal width
    :param table: Table object
    :param delimiters: list of column delimiters
    :param min_width_column: minimal width of a column
    :return: filtered column delimiters list to ensure minimal column width
    """
    # Get column widths
    full_column_delims = [table.x1] + delimiters + [table.x2]
    columns_width = [right - left for left, right in zip(full_column_delims, full_column_delims[1:])]

    # Identify columns that are too small
    small_cols = [max(idx, len(delimiters) - 1)
                  for idx, col_width in enumerate(columns_width) if col_width <= min_width_column]

    return [delim for idx, delim in enumerate(delimiters) if idx not in small_cols]


def get_columns_delimiters(img: np.ndarray, table: Table, vertical_lines: List[Line],
                           min_width_column: int = 15, implicit_columns: bool = False) -> List[int]:
    """
    Identify column delimiters
    :param img: image array
    :param table: Table object
    :param vertical_lines: vertical lines
    :param min_width_column: minimal width of a column
    :param implicit_columns: boolean indicating if implicit columns should be detected
    :return: list of horizontal position of columns
    """
    # Get column delimiters based on vertical lines
    columns_from_line = get_columns_vertical_lines(table, vertical_lines)
    # If some delimiters have been found from vertical lines, return those
    if columns_from_line:
        return columns_from_line

    if not implicit_columns:
        return []

    # Otherwise, add contours for each row of the table
    table_cnts = get_bounding_area_text(img=img,
                                        table=table,
                                        merge_vertically=False)

    # Identify rows with the maximum number of contours / columns
    max_cols = max([len(row.contours) for row in table_cnts.items])

    # If no columns are detected, return empty list
    if max_cols == 0:
        return []

    # Get only rows with the maximum number of contours
    relevant_rows = [row for row in table_cnts.items if len(row.contours) == max_cols]

    # Based on each bounding area, loop on each row to get middle, max and min values of delimiters
    avg_values_delimiter = list()
    min_values_delimiter = list()
    max_values_delimiter = list()
    for row in relevant_rows:
        # Get possible average, max and min values of delimiters from contours
        min_values_delimiter.append([cnt.x2 for cnt in row.contours[:-1]])
        max_values_delimiter.append([cnt.x1 for cnt in row.contours[1:]])
        avg_values_delimiter.append([(cnt_1.x2 + cnt_2.x2) / 2 for cnt_1, cnt_2 in zip(row.contours, row.contours[1:])])

    # Compute minimum, maximum and average values for each delimiter position
    min_values = [max([min_value[idx] for min_value in min_values_delimiter]) for idx in range(max_cols - 1)]
    max_values = [min([max_value[idx] for max_value in max_values_delimiter]) for idx in range(max_cols - 1)]
    avg_values = [round(float(statistics.mean([row[idx] for row in avg_values_delimiter]))) for idx in
                  range(max_cols - 1)]

    # Compute final values according to constraints
    delim_values = [min(max(avg_value, min_value), max_value)
                    for min_value, max_value, avg_value in zip(min_values, max_values, avg_values)]
    delim_values = sorted(list(set(delim_values)))

    # Filter delimiters to guarantee minimal column width
    filtered_delims = get_delimiter_min_width(table=table,
                                              delimiters=delim_values,
                                              min_width_column=min_width_column)

    return filtered_delims


def get_columns_table(img: np.ndarray, table: Table, vertical_lines: List[Line], min_width_column: int = 15,
                      implicit_columns: bool = False) -> Table:
    """
    Identify and create columns in table
    :param img: image array
    :param table: Table object
    :param vertical_lines: vertical lines
    :param min_width_column: minimal width of a column
    :param implicit_columns: boolean indicating if implicit columns should be detected
    :return: table splitted with columns
    """
    # If table already has columns, do not process it
    if table.nb_columns > 1:
        return table

    # Get column delimiters
    col_delimiters = get_columns_delimiters(img=img,
                                            table=table,
                                            vertical_lines=vertical_lines,
                                            min_width_column=min_width_column,
                                            implicit_columns=implicit_columns)

    # If table has only one row and one column, remove it
    if len(col_delimiters) == 0 and table.nb_rows <= 1:
        return None

    # Create columns in table
    table_with_columns = table.split_in_columns(column_delimiters=col_delimiters)

    return table_with_columns
