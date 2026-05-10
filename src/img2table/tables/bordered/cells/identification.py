from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from img2table.tables.bordered.cells._identification import (  # ty:ignore[unresolved-import]
    identify_cells,
)
from img2table.tables.types import Cell

if TYPE_CHECKING:
    from img2table.tables.types import Line


def get_cells_from_lines(horizontal_lines: list[Line], vertical_lines: list[Line]) -> list[Cell]:
    """
    Create dataframe of all possible cells from horizontal and vertical rows
    :param horizontal_lines: list of horizontal rows
    :param vertical_lines: list of vertical rows
    :return: list of detected cells
    """
    # Check for empty rows
    if len(horizontal_lines) * len(vertical_lines) == 0:
        return []

    # Create arrays from horizontal and vertical rows
    h_lines_array = np.array(
        [[line.x1, line.y1, line.x2, line.y2] for line in horizontal_lines], dtype=np.int64
    )
    v_lines_array = np.array(
        [[line.x1, line.y1, line.x2, line.y2] for line in vertical_lines], dtype=np.int64
    )

    # Compute cells
    cells_array = identify_cells(h_lines_arr=h_lines_array, v_lines_arr=v_lines_array)

    return [Cell(*c) for c in cells_array]
