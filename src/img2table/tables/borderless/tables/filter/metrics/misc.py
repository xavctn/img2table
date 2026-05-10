from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.tables.borderless.types import identify_merged_rows

if TYPE_CHECKING:
    from img2table.tables.borderless.tables.filter.model import (
        StructuredSection,
    )


def full_text_score(section: StructuredSection) -> float:
    """
    Identify tables where content occupies the entire span of columns
    :param section: The structured section to compute the metric for.
    :return: score between 0 and 1
    """
    nb_rows, nb_full_rows = 0, 0
    for (start, end), cells in zip(section.cols, section.col_cells, strict=True):
        rows = identify_merged_rows(cnts=cells)

        # Update row number
        nb_rows += len(rows)
        nb_full_rows += len([row for row in rows if row.width >= 0.9 * (end - start)])

    return nb_full_rows / nb_rows if nb_rows else 0.0


def sparsity_score(section: StructuredSection) -> float:
    """
    Compute table sparsity score
    :param section: The structured section to compute the metric for.
    :return: score between 0 and 1
    """
    if section.nb_columns * section.nb_rows == 0:
        return 0.0

    nb_used_cells = 0
    for y_start, y_end in section.row_ranges():
        # Get columns containing an item
        nb_used_cells += sum(
            1
            for col in section.col_cells
            if any(min(cell.y2, y_end) - max(cell.y1, y_start) >= 0.8 * cell.height for cell in col)
        )

    sparsity = 1 - nb_used_cells / (section.nb_columns * section.nb_rows)
    # Reward moderate sparsity and penalize both dense paragraphs and overly empty grids.
    return max(0.0, 1 - abs(sparsity - 0.35) / 0.35)
