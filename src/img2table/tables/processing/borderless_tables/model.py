# coding: utf-8
from dataclasses import dataclass
from typing import List

import numpy as np

from img2table.tables.objects.cell import Cell


@dataclass
class TableLine:
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
    def v_center(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def add(self, c: Cell):
        self.cells.append(c)

    def overlaps(self, other: "TableLine") -> bool:
        # Compute y overlap
        y_top = max(self.y1, other.y1)
        y_bottom = min(self.y2, other.y2)

        return (y_bottom - y_top) / min(self.height, other.height) >= 0.5

    def merge(self, other: "TableLine") -> "TableLine":
        return TableLine(cells=self.cells + other.cells)

    def __eq__(self, other):
        return self.cells == other.cells

    def __hash__(self):
        return hash(f"{self.x1},{self.y1},{self.x2},{self.y2}")


@dataclass
class LineGroup:
    lines: List[TableLine]

    @property
    def x1(self) -> int:
        return min([c.x1 for c in self.lines])

    @property
    def y1(self) -> int:
        return min([c.y1 for c in self.lines])

    @property
    def x2(self) -> int:
        return max([c.x2 for c in self.lines])

    @property
    def y2(self) -> int:
        return max([c.y2 for c in self.lines])

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def size(self) -> int:
        return len(self.lines)

    @property
    def median_line_sep(self) -> float:
        if len(self.lines) <= 1:
            return 0

        # Sort lines
        sorted_lines = sorted(self.lines, key=lambda line: line.y1 + line.y2)

        return np.median([nxt.v_center - prev.v_center for prev, nxt in zip(sorted_lines, sorted_lines[1:])])

    @property
    def median_line_gap(self) -> float:
        if len(self.lines) <= 1:
            return 0

        # Sort lines
        sorted_lines = sorted(self.lines, key=lambda line: line.y1 + line.y2)

        return np.median([nxt.y1 - prev.y2 for prev, nxt in zip(sorted_lines, sorted_lines[1:])])

    def add(self, line: TableLine):
        self.lines.append(line)

    def __hash__(self):
        return hash(repr(self))


@dataclass
class ImageSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    elements: List[Cell] = None
    line_groups: List[LineGroup] = None

    def set_elements(self, elements: List[Cell]):
        self.elements = elements

    def __hash__(self):
        return hash(repr(self))
