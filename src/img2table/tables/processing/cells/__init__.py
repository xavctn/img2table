# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.cells.deduplication import deduplicate_cells
from img2table.tables.processing.cells.identification import get_cells_dataframe


def get_cells(horizontal_lines: List[Line], vertical_lines: List[Line]) -> List[Cell]:
    """
    Identify cells from horizontal and vertical lines
    :param horizontal_lines: list of horizontal lines
    :param vertical_lines: list of vertical lines
    :return: list of all cells in image
    """
    # Create dataframe with cells from horizontal and vertical lines
    df_cells = get_cells_dataframe(horizontal_lines=horizontal_lines,
                                   vertical_lines=vertical_lines)

    # Deduplicate cells
    df_cells_dedup = deduplicate_cells(df_cells=df_cells)

    # Convert to Cell objects
    cells = [Cell(x1=row["x1"], x2=row["x2"], y1=row["y1"], y2=row["y2"])
             for row in df_cells_dedup.collect().to_dicts()]

    return cells
