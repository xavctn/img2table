# coding: utf-8

from dataclasses import dataclass
from typing import Optional, List, OrderedDict

import pandas as pd


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
        return f"Extracted Table(\n\tTitle: {self.title}\n\tBounding Box: x1={self.bbox.x1}, y1={self.bbox.y1}, " \
               f"x2={self.bbox.x2}, y2={self.bbox.y2}\n\tShape: ({len(self.content)}, {len(self.content[0])}))".strip()
