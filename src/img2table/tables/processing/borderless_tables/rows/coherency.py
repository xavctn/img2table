# coding: utf-8
from dataclasses import dataclass
from typing import List

import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow


@dataclass
class LineSpan:
    x_left: int
    x_right: int

    @property
    def center(self) -> float:
        return (self.x_left + self.x_right) / 2


@dataclass
class Alignment:
    type: str
    value: float
    delim_left: Cell
    delim_right: Cell

    @property
    def x_left(self) -> int:
        return self.delim_left.x2

    @property
    def x_right(self) -> int:
        return self.delim_right.x1

    @property
    def width(self) -> int:
        return self.x_right - self.x_left

    def coherent(self, line_span: LineSpan) -> bool:
        if self.type == "left":
            return abs(self.value - line_span.x_left) <= 0.05 * self.width
        elif self.type == "right":
            return abs(self.value - line_span.x_right) <= 0.05 * self.width
        elif self.type == "center":
            return abs(self.value - line_span.center) <= 0.05 * self.width

    def __hash__(self):
        return hash(repr(self))


def identify_content_alignment(delimiter_group: DelimiterGroup, table_rows: List[TableRow]) -> List[Alignment]:
    """
    Identify columns content alignment
    :param delimiter_group: group of column delimiters
    :param table_rows: list of rows corresponding to the delimiter group
    :return: list of content alignment type by column
    """
    # Get column delimiters
    col_delims = sorted(delimiter_group.delimiters, key=lambda d: d.x1)

    alignments = list()
    # For each column, check alignment of content
    for d_left, d_right in zip(col_delims, col_delims[1:]):
        x_left, x_right, y_top, y_bottom = d_left.x2, d_right.x1, min(d_left.y1, d_right.y1), max(d_left.y2, d_right.y2)

        # Get rows that are vertically coherent with delimiters
        coherent_lines = [line for line in table_rows
                          if line.y1 >= max(d_right.y1, d_left.y1)
                          and line.y2 <= min(d_right.y2, d_left.y2)
                          and min(x_right, line.x2) - max(x_left, line.x1) > 0]

        # For each line get span of elements between delimiters
        line_spans = [LineSpan(x_left=min([el.x1 for el in line.cells if el.x1 >= x_left]),
                               x_right=max([el.x2 for el in line.cells if el.x2 <= x_right]))
                      for line in coherent_lines]

        if line_spans:
            # Check for left alignment
            seq = iter(sorted(line_spans, key=lambda ls: ls.x_left))
            left_aligned_gps = [[next(seq)]]
            for ls in seq:
                if ls.x_left - left_aligned_gps[-1][-1].x_left >= 0.05 * (x_right - x_left):
                    left_aligned_gps.append([])
                left_aligned_gps[-1].append(ls)
            # Get largest cluster
            left_aligned_cluster = max(left_aligned_gps, key=len)

            # Check for right alignment
            seq = iter(sorted(line_spans, key=lambda ls: ls.x_right))
            right_aligned_gps = [[next(seq)]]
            for ls in seq:
                if ls.x_right - right_aligned_gps[-1][-1].x_right >= 0.05 * (x_right - x_left):
                    right_aligned_gps.append([])
                right_aligned_gps[-1].append(ls)
            # Get largest cluster
            right_aligned_cluster = max(right_aligned_gps, key=len)

            # Check for center alignment
            seq = iter(sorted(line_spans, key=lambda ls: ls.center))
            center_aligned_gps = [[next(seq)]]
            for ls in seq:
                if ls.center - center_aligned_gps[-1][-1].center >= 0.05 * (x_right - x_left):
                    center_aligned_gps.append([])
                center_aligned_gps[-1].append(ls)
            # Get largest cluster
            center_aligned_cluster = max(center_aligned_gps, key=len)

            # Identify cluster type
            if len(left_aligned_cluster) > max([len(center_aligned_cluster), len(right_aligned_cluster)]):
                alignment = Alignment(type="left",
                                      value=np.mean([ls.x_left for ls in left_aligned_cluster]),
                                      delim_left=d_left,
                                      delim_right=d_right)
            elif len(right_aligned_cluster) > max([len(center_aligned_cluster), len(left_aligned_cluster)]):
                alignment = Alignment(type="right",
                                      value=np.mean([ls.x_right for ls in right_aligned_cluster]),
                                      delim_left=d_left,
                                      delim_right=d_right)
            else:
                alignment = Alignment(type="center",
                                      value=np.mean([ls.center for ls in center_aligned_cluster]),
                                      delim_left=d_left,
                                      delim_right=d_right)
            alignments.append(alignment)

    return alignments


def is_line_coherent(row: TableRow, column_alignments: List[Alignment]) -> bool:
    """
    Identify if a table_rows is coherent based on column alignments
    :param row: TableRow object
    :param column_alignments: list of content alignment type by column
    :return: boolean indicating if a line is coherent based on column alignments
    """
    # Sort columns
    column_alignments = sorted(column_alignments, key=lambda col: col.x_left)

    # Assign each cell to a column interval
    dict_column_elements = dict()
    for cell in row.cells:
        try:
            # Get left and right columns
            left_column = [col for col in column_alignments if col.x_left <= cell.x1
                           and min(col.delim_left.y2, cell.y2) - max(col.delim_left.y1, cell.y1)][-1]
            right_column = [col for col in column_alignments if col.x_right >= cell.x2
                            and min(col.delim_right.y2, cell.y2) - max(col.delim_right.y1, cell.y1)][0]

            if (left_column, right_column) in dict_column_elements:
                dict_column_elements[(left_column, right_column)].append(cell)
            else:
                dict_column_elements[(left_column, right_column)] = [cell]
        except IndexError:
            continue

    # Check coherency of elements with columns
    coherency_checks = list()
    for (col_left, col_right), elements in dict_column_elements.items():
        # Create line span from elements
        line_span = LineSpan(x_left=min([el.x1 for el in elements]),
                             x_right=max([el.x2 for el in elements]))

        if col_left.coherent(line_span=line_span) or col_right.coherent(line_span=line_span):
            coherency_checks.append(True)
        else:
            # Check for center of multiple columns
            cols_center = (col_left.x_left + col_right.x_right) / 2
            coherency = abs(line_span.center - cols_center) <= 0.05 * (col_right.x_right - col_left.x_left)
            coherency_checks.append(coherency)

    # Line is coherent if at least 50% of coherency checks are good
    if coherency_checks:
        return np.mean(coherency_checks) >= 0.5
    return False


def check_extremity_lines(table_rows: List[TableRow], alignments: List[Alignment],
                          median_row_sep: float) -> List[TableRow]:
    """
    Check if extremity rows are coherent with the rest of the table
    :param table_rows: list of rows corresponding
    :param alignments: list of content alignment type by column
    :param median_row_sep: median row separation
    :return: list of table rows with exclusion of incoherent rows
    """
    # Sort rows
    table_rows = sorted(table_rows, key=lambda l: l.y1)

    # Check top rows
    y_min = table_rows[0].y1
    for top_row, bottom_row in zip(table_rows, table_rows[1:]):
        # Check gap between rows and coherency
        gap_coherency = (bottom_row.v_center - top_row.v_center) <= 1.2 * median_row_sep
        row_coherency = is_line_coherent(row=top_row, column_alignments=alignments)
        if not gap_coherency and not row_coherency:
            y_min = bottom_row.y1
        else:
            break

    # Check bottom rows
    table_rows.reverse()
    y_max = table_rows[0].y2
    for bottom_row, top_row in zip(table_rows, table_rows[1:]):
        # Check gap between rows and coherency
        gap_coherency = (bottom_row.v_center - top_row.v_center) <= 1.2 * median_row_sep
        row_coherency = is_line_coherent(row=bottom_row, column_alignments=alignments)
        if not gap_coherency and not row_coherency:
            y_max = top_row.y2
        else:
            break

    # Filter coherent rows
    coherent_rows = [row for row in table_rows if row.y1 >= y_min and row.y2 <= y_max]

    return coherent_rows


def check_coherency_rows(delimiter_group: DelimiterGroup, table_rows: List[TableRow],
                         median_row_sep: float) -> List[TableRow]:
    """
    Check coherency of extreme rows
    :param delimiter_group: group of column delimiters
    :param table_rows: list of rows corresponding to the delimiter group
    :param median_row_sep: median row separation
    :return: list of coherent rows
    """
    if len(table_rows) == 0:
        return table_rows

    # Compute content alignment in columns
    column_alignments = identify_content_alignment(delimiter_group=delimiter_group,
                                                   table_rows=table_rows)

    # Identify coherent rows based on alignment and vertical position
    coherent_rows = check_extremity_lines(table_rows=table_rows,
                                          alignments=column_alignments,
                                          median_row_sep=median_row_sep)

    return coherent_rows
