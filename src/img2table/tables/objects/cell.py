from dataclasses import dataclass

from img2table.tables.objects import TableObject
from img2table.tables.objects.extraction import BBox, TableCell


@dataclass
class Cell(TableObject):
    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float
    content: str | None = None

    @property
    def table_cell(self) -> TableCell:
        bbox = BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
        return TableCell(bbox=bbox, value=self.content)

    def __hash__(self) -> int:
        return hash(repr(self))
