# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.bordered_tables.cells.deduplication import deduplicate_cells
from img2table.tables.processing.bordered_tables.cells.identification import get_cells_dataframe


def get_cells(horizontal_lines: List[Line], vertical_lines: List[Line]) -> List[Cell]:
    """
    Identify cells from horizontal and vertical rows
    :param horizontal_lines: list of horizontal rows
    :param vertical_lines: list of vertical rows
    :return: list of all cells in image
    """
    # Create dataframe with cells from horizontal and vertical rows
    cells = get_cells_dataframe(horizontal_lines=horizontal_lines,
                                vertical_lines=vertical_lines)

    # Deduplicate cells
    dedup_cells = deduplicate_cells(cells=cells)

    return dedup_cells
