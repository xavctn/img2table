# coding:utf-8

from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.cells import get_cells
from img2table.tables.processing.bordered_tables.tables import cluster_to_table
from img2table.tables.processing.borderless_tables.model import ImageSegment
from img2table.tables.processing.borderless_tables.whitespaces import get_whitespaces


def implicit_rows_lines(table: Table, segment: ImageSegment, median_line_sep: float) -> List[Line]:
    """
    Identify lines corresponding to implicit rows
    :param table: table
    :param segment: ImageSegment used for whitespaces computation
    :param median_line_sep: median row separation
    :return: list of lines corresponding to implicit rows
    """
    # Horizontal whitespaces
    h_ws = get_whitespaces(segment=segment,
                           vertical=False,
                           min_width=median_line_sep // 4,
                           pct=1)

    # Identify created lines
    created_lines = list()
    for ws in h_ws:
        if not any([line for line in table.lines if ws.y1 <= line.y1 <= ws.y2 and line.horizontal]):
            created_lines.append(Line(x1=table.x1,
                                      y1=(ws.y1 + ws.y2) // 2,
                                      x2=table.x2,
                                      y2=(ws.y1 + ws.y2) // 2))

    return created_lines


def implicit_columns_lines(table: Table, segment: ImageSegment, char_length: float) -> List[Line]:
    """
    Identify lines corresponding to implicit columns
    :param table: table
    :param segment: ImageSegment used for whitespaces computation
    :param char_length: average character length
    :return: list of lines corresponding to implicit columns
    """
    # Vertical whitespaces
    v_ws = get_whitespaces(segment=segment,
                           vertical=True,
                           min_width=char_length,
                           pct=1)

    # Identify created lines
    created_lines = list()
    for ws in v_ws:
        if not any([line for line in table.lines if ws.x1 <= line.x1 <= ws.x2 and line.vertical]):
            created_lines.append(Line(x1=(ws.x1 + ws.x2) // 2,
                                      y1=table.y1,
                                      x2=(ws.x1 + ws.x2) // 2,
                                      y2=table.y2))

    return created_lines


def implicit_content(table: Table, contours: List[Cell], char_length: float, median_line_sep: float,
                     implicit_rows: bool = False, implicit_columns: bool = False) -> Table:
    """
    Identify implicit content in table
    :param table: Table object
    :param contours: image contours
    :param char_length: average character length
    :param median_line_sep: median row separation
    :param implicit_rows: boolean indicating if implicit rows should be detected
    :param implicit_columns: boolean indicating if implicit columns should be detected
    :return: Table with implicit content detected
    """
    if not implicit_rows and not implicit_columns:
        return table

    # Get table contours and create corresponding segment
    tb_contours = [c for c in contours
                   if c.x1 >= table.x1 and c.x2 <= table.x2
                   and c.y1 >= table.y1 and c.y2 <= table.y2]
    segment = ImageSegment(x1=table.x1, y1=table.y1, x2=table.x2, y2=table.y2,
                           elements=tb_contours)

    # Create new lines
    lines = table.lines
    if implicit_rows:
        lines += implicit_rows_lines(table=table, segment=segment, median_line_sep=median_line_sep)
    if implicit_columns:
        lines += implicit_columns_lines(table=table, segment=segment, char_length=char_length)

    # Create
    cells = get_cells(horizontal_lines=[line for line in lines if line.horizontal],
                      vertical_lines=[line for line in lines if line.vertical])

    return cluster_to_table(cluster_cells=cells, elements=tb_contours, borderless=False)
