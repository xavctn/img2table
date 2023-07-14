# coding: utf-8

from dataclasses import dataclass
from typing import Optional, List, OrderedDict, NamedTuple, Tuple

import pandas as pd
from xlsxwriter.format import Format
from xlsxwriter.worksheet import Worksheet


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class TableCell:
    bbox: BBox
    value: Optional[str]

    def __hash__(self):
        return hash(repr(self))


class CellPosition(NamedTuple):
    cell: TableCell
    row: int
    col: int


def create_all_rectangles(cell_positions: List[CellPosition]) -> List[Tuple]:
    """
    Create all possible rectangles from list of cell positions
    :param cell_positions: list of cell positions
    :return: list of tuples representing rectangle coordinates
    """
    # Get bounding coordinates
    min_col = min(map(lambda x: x.col, cell_positions))
    max_col = max(map(lambda x: x.col, cell_positions))
    min_row = min(map(lambda x: x.row, cell_positions))
    max_row = max(map(lambda x: x.row, cell_positions))

    # Get largest rectangle fully covered by cell positions
    largest_area, area_coords, area_cell_pos = 0, None, None
    for col_left in range(min_col, max_col + 1):
        for col_right in range(col_left, max_col + 1):
            for top_row in range(min_row, max_row + 1):
                for bottom_row in range(top_row, max_row + 1):
                    # Get matching cell positions
                    matching_cell_pos = [cp for cp in cell_positions if col_left <= cp.col <= col_right
                                         and top_row <= cp.row <= bottom_row]

                    # Check if the rectangle is fully covered
                    fully_covered = len(matching_cell_pos) == (col_right - col_left + 1) * (bottom_row - top_row + 1)

                    # If rectangle is the largest, update values
                    if fully_covered and (len(matching_cell_pos) > largest_area):
                        largest_area = len(matching_cell_pos)
                        area_cell_pos = matching_cell_pos
                        area_coords = (col_left, top_row, col_right, bottom_row)

    # Get remaining cell positions
    remaining_cell_positions = [cp for cp in cell_positions if cp not in area_cell_pos]

    if remaining_cell_positions:
        # Get remaining rectangles
        return [area_coords] + create_all_rectangles(remaining_cell_positions)
    else:
        # Return coordinates
        return [area_coords]


@dataclass
class ExtractedTable:
    bbox: BBox
    title: Optional[str]
    content: OrderedDict[int, List[TableCell]]

    @property
    def df(self) -> pd.DataFrame:
        """
        Create pandas DataFrame representation of the table
        :return: pandas DataFrame containing table data
        """
        values = [[cell.value for cell in row] for k, row in self.content.items()]
        return pd.DataFrame(values)

    def _to_worksheet(self, sheet: Worksheet, cell_fmt: Optional[Format] = None):
        """
        Populate xlsx worksheet with table data
        :param sheet: xlsxwriter Worksheet
        :param cell_fmt: xlsxwriter cell format
        """
        # Group cells based on hash (merged cells are duplicated over multiple rows/columns in content)
        dict_cells = dict()
        for id_row, row in self.content.items():
            for id_col, cell in enumerate(row):
                cell_pos = CellPosition(cell=cell, row=id_row, col=id_col)
                dict_cells[hash(cell)] = dict_cells.get(hash(cell), []) + [cell_pos]

        # Write all cells to sheet
        for c in dict_cells.values():
            if len(c) == 1:
                cell_pos = c.pop()
                sheet.write(cell_pos.row, cell_pos.col, cell_pos.cell.value, cell_fmt)
            else:
                # Get all rectangles
                for rect in create_all_rectangles(cell_positions=c):
                    col_left, top_row, col_right, bottom_row = rect
                    # Case of merged cells
                    sheet.merge_range(first_row=top_row,
                                      first_col=col_left,
                                      last_row=bottom_row,
                                      last_col=col_right,
                                      data=c[0].cell.value,
                                      cell_format=cell_fmt)

        # Autofit worksheet
        sheet.autofit()

    def html_repr(self, title: Optional[str] = None) -> str:
        """
        Create HTML representation of the table
        :param title: title of HTML paragraph
        :return: HTML string
        """
        html = f"""{rf'<h3 style="text-align: center">{title}</h3>' if title else ''}
                   <p style=\"text-align: center\">
                       <b>Title:</b> {self.title or 'No title detected'}<br>
                       <b>Bounding box:</b> x1={self.bbox.x1}, y1={self.bbox.y1}, x2={self.bbox.x2}, y2={self.bbox.y2}
                   </p>
                   <div align=\"center\">{self.df.to_html().replace("None", "")}</div>
                   <hr>
                """
        return html

    def __repr__(self):
        return f"ExtractedTable(title={self.title}, bbox=({self.bbox.x1}, {self.bbox.y1}, {self.bbox.x2}, " \
               f"{self.bbox.y2}),shape=({len(self.content)}, {len(self.content[0])}))".strip()
