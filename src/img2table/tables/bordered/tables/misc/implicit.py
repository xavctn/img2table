from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

from img2table.tables.bordered.cells import get_cells
from img2table.tables.bordered.tables import cluster_to_table
from img2table.tables.borderless.types import (
    compute_whitespaces,
    identify_merged_rows,
)
from img2table.tables.common import compute_row_ranges
from img2table.tables.types import Line

if TYPE_CHECKING:
    from img2table.tables.types import Cell, Table


def implicit_rows_lines(table: Table, contours: list[Cell]) -> list[Line]:
    """
    Identify lines corresponding to implicit rows
    :param table: table
    :param contours: list of contours
    :return: list of lines corresponding to implicit rows
    """
    # Compute merged rows
    merged_rows = identify_merged_rows(cnts=contours)

    # Identify new lines
    new_lines = []
    for tb_row in table.rows:
        if not tb_row.v_consistent:
            # Skip rows containing merged cells
            continue

        # Get corresponding merged rows and compute row ranges
        row_ranges = compute_row_ranges(
            rows=[row for row in merged_rows if row.y1 >= tb_row.y1 and row.y2 <= tb_row.y2],
            y_min=tb_row.y1,
            y_max=tb_row.y2,
        )

        # Identify new lines
        relevant_y_vals = {
            bound
            for bounds in row_ranges
            for bound in bounds
            if bound not in {tb_row.y1, tb_row.y2}
        }
        new_lines += [
            Line(x1=tb_row.x1, y1=y, x2=tb_row.x2, y2=y, thickness=1) for y in relevant_y_vals
        ]

    return new_lines


def implicit_columns_lines(table: Table, contours: list[Cell], char_length: float) -> list[Line]:
    """
    Identify lines corresponding to implicit columns
    :param table: table
    :param contours: list of contours
    :param char_length: average character length
    :return: list of lines corresponding to implicit columns
    """
    # Get vertical whitespaces
    v_ws = compute_whitespaces(
        items=contours, min_width=char_length, x_min=table.x1, x_max=table.x2
    )

    # Identify columns with merged cells and their span
    horizontal_merged_span = {
        (prv_cell.x1, prv_cell.x2)
        for row in table.rows
        for prv_cell, nxt_cell in pairwise(row.cells)
        if prv_cell == nxt_cell
    }

    # Identify created lines
    return [
        Line(x1=(ws.start + ws.end) // 2, y1=table.y1, x2=(ws.start + ws.end) // 2, y2=table.y2)
        for ws in v_ws
        if not ws.start_bound
        and not ws.end_bound
        and not any(line for line in table.lines if ws.start <= line.x1 <= ws.end and line.vertical)
        and not any(
            1
            for (start, end) in horizontal_merged_span
            if max(ws.end, end) - min(ws.start, start) > 0
        )
    ]


def implicit_content(
    table: Table,
    contours: list[Cell],
    char_length: float,
    implicit_rows: bool = False,
    implicit_columns: bool = False,
) -> Table:
    """
    Identify implicit content in table
    :param table: Table object
    :param contours: image contours
    :param char_length: average character length
    :param implicit_rows: boolean indicating if implicit rows should be detected
    :param implicit_columns: boolean indicating if implicit columns should be detected
    :return: Table with implicit content detected
    """
    if not implicit_rows and not implicit_columns:
        return table

    # Get table contours and create corresponding segment
    tb_contours = [
        c
        for c in contours
        if c.x1 >= table.x1 and c.x2 <= table.x2 and c.y1 >= table.y1 and c.y2 <= table.y2
    ]

    # Create new lines
    new_lines = []
    if implicit_rows:
        new_lines += implicit_rows_lines(table=table, contours=tb_contours)
    if implicit_columns:
        new_lines += implicit_columns_lines(
            table=table, contours=tb_contours, char_length=char_length
        )

    if len(new_lines) == 0:
        # Nothing added
        return table

    # Create new cells
    cells = get_cells(
        horizontal_lines=[line for line in table.lines + new_lines if line.horizontal],
        vertical_lines=[line for line in table.lines + new_lines if line.vertical],
    )

    # Compute updated table
    return cluster_to_table(cluster_cells=cells, elements=tb_contours)
