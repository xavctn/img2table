from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.tables.bordered.cells import get_cells
from img2table.tables.bordered.tables import get_tables
from img2table.tables.bordered.tables.misc.implicit import implicit_content

if TYPE_CHECKING:
    from img2table.tables.types import Cell, Line, Table


def extract_bordered_tables(
    contours: list[Cell],
    horizontal_lines: list[Line],
    vertical_lines: list[Line],
    char_length: float,
    implicit_rows: bool = False,
    implicit_columns: bool = False,
) -> list[Table]:
    """
    Identify bordered tables in image.
    :param contours: List of detected contours in image.
    :param horizontal_lines: List of horizontal lines.
    :param vertical_lines: List of vertical lines.
    :param char_length: Character length in pixels.
    :param implicit_rows: Whether to detect implicit rows.
    :param implicit_columns: Whether to detect implicit columns.
    :return: List of bordered tables.
    """
    # Create cells from rows
    cells = get_cells(horizontal_lines=horizontal_lines, vertical_lines=vertical_lines)

    # Create tables from rows
    tables = get_tables(
        cells=cells,
        elements=contours,
        lines=horizontal_lines + vertical_lines,
        char_length=char_length,
    )

    # If necessary, detect implicit rows
    return [
        implicit_content(
            table=table,
            contours=contours,
            char_length=char_length,
            implicit_rows=implicit_rows,
            implicit_columns=implicit_columns,
        )
        for table in tables
    ]
