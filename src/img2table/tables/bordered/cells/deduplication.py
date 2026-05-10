from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from img2table.tables.types import Cell


def deduplicate_cells(cells: list[Cell]) -> list[Cell]:
    """
    Deduplicate nested cells in order to keep the smallest ones
    :param cells: list of cells
    :return: cells after deduplication of the nested ones
    """
    # Create array of cell coverages
    x_max, y_max = max((c.x2 for c in cells), default=0), max((c.y2 for c in cells), default=0)
    coverage_array = np.ones((y_max, x_max), dtype=np.uint8)

    dedup_cells = []
    for c in sorted(cells, key=lambda c: c.area):
        cropped = coverage_array[c.y1 : c.y2, c.x1 : c.x2]
        # If cell has at least 25% of its area not covered, add it
        if np.sum(cropped) >= 0.25 * c.area:
            dedup_cells.append(c)
            coverage_array[c.y1 : c.y2, c.x1 : c.x2] = 0

    return dedup_cells
