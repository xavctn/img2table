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
        values = [[cell.value for cell in row] for k, row in self.content.items()]
        return pd.DataFrame(values)
