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
        self: "CoordinateProvider", margin: int = 0, height_margin: int = 0, width_margin: int = 0
    ) -> tuple[int, int, int, int]:
        """
        Return bounding box corresponding to the object
        :param margin: general margin used for the bounding box
        :param height_margin: vertical margin used for the bounding box
        :param width_margin: horizontal margin used for the bounding box
        :return: tuple representing a bounding box
        """
        # Apply margin on bbox
        if margin != 0:
            bbox = (self.x1 - margin, self.y1 - margin, self.x2 + margin, self.y2 + margin)
        else:
            bbox = (
                self.x1 - width_margin,
                self.y1 - height_margin,
                self.x2 + width_margin,
                self.y2 + height_margin,
            )

        return bbox

    @property
    def height(self: "CoordinateProvider") -> int:
        return self.y2 - self.y1

    @property
    def width(self: "CoordinateProvider") -> int:
        return self.x2 - self.x1

    @property
    def area(self) -> int:
        return self.height * self.width
