from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class CoordinateProvider(Protocol):
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class TableObject:
    def bbox(
        self: CoordinateProvider, margin: int = 0
    ) -> tuple[int, int, int, int]:
        """
        Return bounding box corresponding to the object
        :param margin: general margin used for the bounding box
        :return: tuple representing a bounding box
        """
        return (self.x1 - margin, self.y1 - margin, self.x2 + margin, self.y2 + margin)

    @property
    def height(self: CoordinateProvider) -> int:
        return self.y2 - self.y1

    @property
    def width(self: CoordinateProvider) -> int:
        return self.x2 - self.x1

    @property
    def area(self) -> int:
        return self.height * self.width
