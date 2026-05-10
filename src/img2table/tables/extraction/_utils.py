from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xlsxwriter.format import Format
    from xlsxwriter.worksheet import Worksheet

    from img2table.tables.extraction import ExtractedTable, TableCell


@dataclass
class CellPosition:
    cell: TableCell
    row: int
    col: int


@dataclass
class CellSpan:
    top_row: int
    bottom_row: int
    col_left: int
    col_right: int
    value: str | None

    @property
    def colspan(self) -> int:
        return self.col_right - self.col_left + 1

    @property
    def rowspan(self) -> int:
        return self.bottom_row - self.top_row + 1

    @property
    def html_value(self) -> str:
        if self.value is not None:
            return escape(self.value).replace("\n", "<br>")
        return ""

    @property
    def html(self) -> str:
        td = "<td"
        if self.colspan > 1:
            td += f' colspan="{self.colspan}"'
        if self.rowspan > 1:
            td += f' rowspan="{self.rowspan}"'
        return f"{td}>{self.html_value}</td>"

    def html_cell_span(self) -> list[CellSpan]:
        if self.colspan > 1 and self.rowspan > 1:
            # Check largest coordinate and split
            if self.colspan > self.rowspan:
                return [
                    CellSpan(
                        top_row=row_idx,
                        bottom_row=row_idx,
                        col_left=self.col_left,
                        col_right=self.col_right,
                        value=self.value,
                    )
                    for row_idx in range(self.top_row, self.bottom_row + 1)
                ]
            return [
                CellSpan(
                    top_row=self.top_row,
                    bottom_row=self.bottom_row,
                    col_left=col_idx,
                    col_right=col_idx,
                    value=self.value,
                )
                for col_idx in range(self.col_left, self.col_right + 1)
            ]

        return [self]


def group_cell_positions(table: ExtractedTable) -> list[list[CellPosition]]:
    """
    Create a list of cell positions for each cell in the table.
    :param table: ExtractedTable instance
    :return: list of cell positions for each cell in the table
    """
    dict_cells: dict[TableCell, list[CellPosition]] = {}
    for id_row, row in table.content.items():
        for id_col, cell in enumerate(row):
            cell_pos = CellPosition(cell=cell, row=id_row, col=id_col)
            dict_cells[cell] = [*dict_cells.get(cell, []), cell_pos]

    return list(dict_cells.values())


def _find_largest_rectangle(
    pos_set: set[tuple[int, int]], min_row: int, max_row: int, min_col: int, max_col: int
) -> tuple[int, int, int, int]:
    """
    Find the largest fully-covered rectangle using the histogram DP approach
    :param pos_set: set of (row, col) positions occupied by cell positions
    :param min_row: minimum row index
    :param max_row: maximum row index
    :param min_col: minimum column index
    :param max_col: maximum column index
    :return: (col_left, top_row, col_right, bottom_row) in absolute coordinates
    """
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1
    heights = [0] * cols
    best_area = 0
    best = (min_col, min_row, min_col, min_row)

    for r in range(rows):
        for c in range(cols):
            heights[c] = (heights[c] + 1) if (r + min_row, c + min_col) in pos_set else 0

        # Largest rectangle in histogram via stack
        stack: list[tuple[int, int]] = []  # (height, start_col)
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][0] > h:
                height, start_c = stack.pop()
                area = height * (c - start_c)
                if area > best_area:
                    best_area = area
                    best = (
                        start_c + min_col,
                        r - height + 1 + min_row,
                        c - 1 + min_col,
                        r + min_row,
                    )
                start = start_c
            stack.append((h, start))

    return best


def create_all_rectangles(cell_positions: list[CellPosition]) -> list[CellSpan]:
    """
    Create all possible rectangles from list of cell positions
    :param cell_positions: list of cell positions
    :return: list of CellSpan objects representing rectangle coordinates
    """
    if not cell_positions:
        return []

    # Compute the largest rectangle that covers all cell positions
    col_left, top_row, col_right, bottom_row = _find_largest_rectangle(
        pos_set={(cp.row, cp.col) for cp in cell_positions},
        min_row=min(cp.row for cp in cell_positions),
        max_row=max(cp.row for cp in cell_positions),
        min_col=min(cp.col for cp in cell_positions),
        max_col=max(cp.col for cp in cell_positions),
    )
    cell_span = CellSpan(
        col_left=col_left,
        top_row=top_row,
        col_right=col_right,
        bottom_row=bottom_row,
        value=cell_positions[0].cell.value,
    )

    # Compute covered cells and remaining positions
    covered = {
        (r, c) for r in range(top_row, bottom_row + 1) for c in range(col_left, col_right + 1)
    }
    remaining = [cp for cp in cell_positions if (cp.row, cp.col) not in covered]

    if remaining:
        return [cell_span, *create_all_rectangles(remaining)]
    return [cell_span]


def build_html(table: ExtractedTable) -> str:
    """
    Create HTML representation of the table
    :param table: ExtractedTable instance
    :return: HTML table
    """
    from bs4 import BeautifulSoup

    # Get list of cell spans
    cell_span_list = [
        cell_span
        for cells in group_cell_positions(table=table)
        for cell_span in create_all_rectangles(cell_positions=cells)
    ]
    cell_span_list = [span for cell_span in cell_span_list for span in cell_span.html_cell_span()]

    # Create HTML rows
    rows_html = []
    for row_idx in range(len(table.content)):
        # Get cells in row
        row_cells = sorted(
            [cell_span for cell_span in cell_span_list if cell_span.top_row == row_idx],
            key=lambda cs: cs.col_left,
        )
        html_row = "<tr>" + "".join([cs.html for cs in row_cells]) + "</tr>"
        rows_html.append(html_row)

    # Create HTML table
    table_html = "<table>" + "".join(rows_html) + "</table>"

    return BeautifulSoup(table_html, "html.parser").prettify().strip()


def build_worksheet(
    table: ExtractedTable, sheet: Worksheet, cell_fmt: Format | None = None
) -> None:
    """
    Populate xlsx worksheet with table data
    :param table: ExtractedTable instance
    :param sheet: xlsxwriter Worksheet
    :param cell_fmt: xlsxwriter cell format
    """
    # Write all cells to sheet
    for c in group_cell_positions(table=table):
        if len(c) == 1:
            cell_pos = c[0]
            sheet.write(cell_pos.row, cell_pos.col, cell_pos.cell.value, cell_fmt)
        else:
            # Get all rectangles
            for cell_span in create_all_rectangles(cell_positions=c):
                # Case of merged cells
                sheet.merge_range(
                    first_row=cell_span.top_row,
                    first_col=cell_span.col_left,
                    last_row=cell_span.bottom_row,
                    last_col=cell_span.col_right,
                    data=cell_span.value,
                    cell_format=cell_fmt,
                )

    # Autofit worksheet
    sheet.autofit()
