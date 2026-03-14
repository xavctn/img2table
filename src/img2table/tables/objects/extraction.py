from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import pandas as pd
    from xlsxwriter.format import Format
    from xlsxwriter.worksheet import Worksheet


@dataclass
class BBox:
    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float


@dataclass
class TableCell:
    bbox: BBox
    value: str | None

    def __hash__(self) -> int:
        return hash(repr(self))


class CellPosition(NamedTuple):
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
            return self.value.replace("\n", "<br>")
        return ""

    @property
    def html(self) -> str:
        return f'<td colspan="{self.colspan}" rowspan="{self.rowspan}">{self.html_value}</td>'

    def html_cell_span(self) -> list["CellSpan"]:
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


@dataclass
class ExtractedTable:
    bbox: BBox
    title: str | None
    content: OrderedDict[int, list[TableCell]]

    @property
    def df(self) -> "pd.DataFrame":
        """
        Create pandas DataFrame representation of the table
        :return: pandas DataFrame containing table data
        """
        import pandas as pd

        values = [[cell.value for cell in row] for k, row in self.content.items()]
        return pd.DataFrame(values)

    @property
    def html(self) -> str:
        """
        Create HTML representation of the table
        :return: HTML table
        """
        from bs4 import BeautifulSoup

        # Group cells based on hash (merged cells are duplicated over multiple rows/columns in content)
        dict_cells = {}
        for id_row, row in self.content.items():
            for id_col, cell in enumerate(row):
                cell_pos = CellPosition(cell=cell, row=id_row, col=id_col)
                dict_cells[hash(cell)] = [*dict_cells.get(hash(cell), []), cell_pos]

        # Get list of cell spans
        cell_span_list = [
            cell_span
            for _, cells in dict_cells.items()
            for cell_span in create_all_rectangles(cell_positions=cells)
        ]
        cell_span_list = [
            span for cell_span in cell_span_list for span in cell_span.html_cell_span()
        ]

        # Create HTML rows
        rows_html = []
        for row_idx in range(len(self.content)):
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

    def _to_worksheet(self, sheet: "Worksheet", cell_fmt: "Format | None" = None) -> None:
        """
        Populate xlsx worksheet with table data
        :param sheet: xlsxwriter Worksheet
        :param cell_fmt: xlsxwriter cell format
        """
        # Group cells based on hash (merged cells are duplicated over multiple rows/columns in content)
        dict_cells = {}
        for id_row, row in self.content.items():
            for id_col, cell in enumerate(row):
                cell_pos = CellPosition(cell=cell, row=id_row, col=id_col)
                dict_cells[hash(cell)] = [*dict_cells.get(hash(cell), []), cell_pos]

        # Write all cells to sheet
        for c in dict_cells.values():
            if len(c) == 1:
                cell_pos = c.pop()
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

    def html_repr(self, title: str | None = None) -> str:
        """
        Create HTML representation of the table
        :param title: title of HTML paragraph
        :return: HTML string
        """
        html = f"""{rf'<h3 style="text-align: center">{title}</h3>' if title else ""}
                   <p style=\"text-align: center\">
                       <b>Title:</b> {self.title or "No title detected"}<br>
                       <b>Bounding box:</b> x1={self.bbox.x1}, y1={self.bbox.y1}, x2={self.bbox.x2}, y2={self.bbox.y2}
                   </p>
                   <div align=\"center\">{self.df.to_html().replace("None", "")}</div>
                   <hr>
                """
        return html  # noqa: RET504

    def __repr__(self) -> str:
        return (
            f"ExtractedTable(title={self.title}, bbox=({self.bbox.x1}, {self.bbox.y1}, {self.bbox.x2}, "
            f"{self.bbox.y2}),shape=({len(self.content)}, {len(self.content[0])}))".strip()
        )
