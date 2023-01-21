# coding: utf-8

from dataclasses import dataclass
from typing import Optional, List, OrderedDict, NamedTuple

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
                # Case of merged cells
                sheet.merge_range(first_row=min(map(lambda x: x.row, c)),
                                  first_col=min(map(lambda x: x.col, c)),
                                  last_row=max(map(lambda x: x.row, c)),
                                  last_col=max(map(lambda x: x.col, c)),
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
                   <div align=\"center\">{self.df.to_html()}</div>
                   <hr>
                """
        return html

    def __repr__(self):
        return f"ExtractedTable(title={self.title}, bbox=({self.bbox.x1}, {self.bbox.y1}, {self.bbox.x2}, " \
               f"{self.bbox.y2}),shape=({len(self.content)}, {len(self.content[0])}))".strip()
