from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from img2table.tables.objects import TableObject


@dataclass
class Line(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: int = 1

    @property
    def angle(self) -> float:
        delta_x = self.x2 - self.x1
        delta_y = self.y2 - self.y1

        return math.atan2(delta_y, delta_x) * 180 / np.pi

    @property
    def length(self) -> float:
        return np.sqrt(self.height**2 + self.width**2)

    @property
    def horizontal(self) -> bool:
        return self.angle % 180 == 0

    @property
    def vertical(self) -> bool:
        return self.angle % 180 == 90

    def reprocess(self) -> Line:
        # Reallocate coordinates in proper order
        _x1 = min(self.x1, self.x2)
        _x2 = max(self.x1, self.x2)
        _y1 = min(self.y1, self.y2)
        _y2 = max(self.y1, self.y2)
        self.x1, self.x2, self.y1, self.y2 = _x1, _x2, _y1, _y2

        # Correct "almost" horizontal or vertical rows
        if abs(self.angle) <= 5:
            y_val = round((self.y1 + self.y2) / 2)
            self.y2 = self.y1 = y_val
        elif abs(self.angle - 90) <= 5:
            x_val = round((self.x1 + self.x2) / 2)
            self.x2 = self.x1 = x_val

        return self

    def __hash__(self) -> int:
        return hash(repr(self))
