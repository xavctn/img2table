from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.tables.bordered.cells.deduplication import deduplicate_cells
from img2table.tables.bordered.cells.identification import get_cells_from_lines

if TYPE_CHECKING:
    from img2table.tables.types import Cell, Line


def get_cells(horizontal_lines: list[Line], vertical_lines: list[Line]) -> list[Cell]:
    """
    Identify cells from horizontal and vertical rows
    :param horizontal_lines: list of horizontal rows
    :param vertical_lines: list of vertical rows
    :return: list of all cells in image
    """
    # Create dataframe with cells from horizontal and vertical rows
    cells = get_cells_from_lines(horizontal_lines=horizontal_lines, vertical_lines=vertical_lines)

    # Deduplicate cells
    return deduplicate_cells(cells=cells)
