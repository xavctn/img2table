# coding: utf-8
from dataclasses import dataclass

from img2table.tables.objects import TableObject
from img2table.tables.objects.line import Line


@dataclass
class Cell(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_h_lines(cls, line_1: Line, line_2: Line, minimal: bool = False) -> "Cell":
        """
        Generate cell from two lines
        :param line_1: first line
        :param line_2: second line
        :param minimal: boolean indicating if cell should be as the larger intersection of both lines
        :return: Cell object
        """
        # Compute coordinates
        x1 = max(line_1.x1, line_2.x1) if minimal else min(line_1.x1, line_2.x1)
        x2 = min(line_1.x2, line_2.x2) if minimal else max(line_1.x2, line_2.x2)
        y1 = min(line_1.y1, line_2.y1)
        y2 = max(line_1.y2, line_2.y2)

        return cls(x1=x1, x2=x2, y1=y1, y2=y2)
