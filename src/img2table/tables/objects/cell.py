# coding: utf-8
from dataclasses import dataclass

from img2table.tables.objects import TableObject
from img2table.tables.objects.extraction import TableCell, BBox


@dataclass
class Cell(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int
    content: str = None

    @property
    def table_cell(self) -> TableCell:
        bbox = BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
        return TableCell(bbox=bbox, value=self.content)

    def __hash__(self):
        return hash(repr(self))
