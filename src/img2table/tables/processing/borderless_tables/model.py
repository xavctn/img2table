# coding: utf-8
from dataclasses import dataclass, field
from typing import List

from img2table.tables.objects.cell import Cell


@dataclass
class Whitespace:
    cells: List[Cell]

    @property
    def x1(self) -> int:
        return min([c.x1 for c in self.cells])

    @property
    def y1(self) -> int:
        return min([c.y1 for c in self.cells])

    @property
    def x2(self) -> int:
        return max([c.x2 for c in self.cells])

    @property
    def y2(self) -> int:
        return max([c.y2 for c in self.cells])

    @property
    def width(self) -> int:
        return sum([c.width for c in self.cells])

    @property
    def height(self) -> int:
        return sum([c.height for c in self.cells])

    @property
    def area(self) -> int:
        return sum([c.area for c in self.cells])

    @property
    def continuous(self) -> bool:
        return len(self.cells) == 1

    def flipped(self) -> "Whitespace":
        return Whitespace(cells=[Cell(x1=c.y1, y1=c.x1, x2=c.y2, y2=c.x2) for c in self.cells])

    def reshape(self, x1: int, y1: int, x2: int, y2: int) -> "Whitespace":
        return Whitespace(cells=[Cell(x1=max(c.x1, x1),
                                      y1=max(c.y1, y1),
                                      x2=min(c.x2, x2),
                                      y2=min(c.y2, y2))
                                 for c in self.cells])

    def __contains__(self, item: "Whitespace") -> bool:
        return self.x1 <= item.x1 and self.y1 <= item.y1 and self.x2 >= item.x2 and self.y2 >= item.y2

    def __hash__(self):
        return hash(repr(self))


@dataclass
class ImageSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    elements: List[Cell] = None
    whitespaces: List[Whitespace] = None
    position: int = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def set_elements(self, elements: List[Cell]):
        self.elements = elements

    def set_whitespaces(self, whitespaces: List[Whitespace]):
        self.whitespaces = whitespaces

    def __hash__(self):
        return hash(repr(self))


@dataclass
class TableSegment:
    table_areas: List[ImageSegment]

    @property
    def x1(self) -> int:
        return min([tb_area.x1 for tb_area in self.table_areas])

    @property
    def y1(self) -> int:
        return min([tb_area.y1 for tb_area in self.table_areas])

    @property
    def x2(self) -> int:
        return max([tb_area.x2 for tb_area in self.table_areas])

    @property
    def y2(self) -> int:
        return max([tb_area.y2 for tb_area in self.table_areas])

    @property
    def elements(self) -> List[Cell]:
        return [el for tb_area in self.table_areas for el in tb_area.elements]

    @property
    def whitespaces(self) -> List[Whitespace]:
        return [ws for tb_area in self.table_areas for ws in tb_area.whitespaces]


@dataclass
class DelimiterGroup:
    delimiters: List[Cell]
    elements: List[Cell] = field(default_factory=lambda: [])

    @property
    def x1(self) -> int:
        if self.delimiters:
            return min([d.x1 for d in self.delimiters])
        return 0

    @property
    def y1(self) -> int:
        if self.delimiters:
            return min([d.y1 for d in self.delimiters])
        return 0

    @property
    def x2(self) -> int:
        if self.delimiters:
            return max([d.x2 for d in self.delimiters])
        return 0

    @property
    def y2(self) -> int:
        if self.delimiters:
            return max([d.y2 for d in self.delimiters])
        return 0

    @property
    def bbox(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def add(self, delim: Cell):
        self.delimiters.append(delim)

    def __eq__(self, other):
        if isinstance(other, DelimiterGroup):
            try:
                assert set(self.delimiters) == set(other.delimiters)
                assert set(self.elements) == set(other.elements)
                return True
            except AssertionError:
                return False
        return False
