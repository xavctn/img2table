from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from img2table.tables.borderless.types import identify_merged_rows

if TYPE_CHECKING:
    from img2table.tables.borderless.tables.filter.model import (
        StructuredSection,
    )
    from img2table.tables.types import Cell


def _median_absolute_difference(values: list[float]) -> float:
    """
    Compute MAD of series
    :param values: list of values
    :return: series MAD
    """
    median = np.median(values)
    abs_diffs = [abs(x - median) for x in values]
    return float(np.median(abs_diffs) + np.mean(abs_diffs)) / 2


def _col_alignment_score(col: list[Cell], char_length: float, width: int) -> float | None:
    """
    Compute alignment score of a specific column
    :param col: list of cells in the column
    :param char_length: average character length
    :param width: width of the image
    :return: alignment score of the column
    """
    # Check column width
    if (
        max((cnt.x2 for cnt in col), default=0) - min((cnt.x1 for cnt in col), default=0)
        < 3 * char_length
    ) and len(col) < 3:
        return 0.0

    # Compute merged rows
    rows = identify_merged_rows(cnts=col)

    if len(rows) < 2:
        return None

    alignments = {"left_alignment": [], "center_alignment": [], "right_alignment": []}
    for cell in rows:
        alignments["left_alignment"].append(cell.x1)
        alignments["center_alignment"].append((cell.x1 + cell.x2) / 2)
        alignments["right_alignment"].append(cell.x2)

    # Compute deviations
    deviations = {
        "left_alignment": _median_absolute_difference(alignments["left_alignment"]),
        "center_alignment": _median_absolute_difference(alignments["center_alignment"]),
        "right_alignment": _median_absolute_difference(alignments["right_alignment"]),
    }

    # Get alignment with lowest deviation
    best_alignment = min(deviations, key=lambda k: deviations[k])

    if best_alignment == "center_alignment":
        score = 1 - deviations["center_alignment"] / (0.02 * width)
    elif best_alignment == "left_alignment":
        # Check that main alignment is at the left bound
        values, deviation = alignments["left_alignment"], deviations["left_alignment"]
        if (np.median(values) - min(values)) / (max(values) - min(values) + 10e-6) <= 0.05:
            score = 1 - deviation / (0.02 * width)
        else:
            # Apply penalization
            score = 1 - (2 * deviation) / (0.02 * width)
    else:
        # Check that main alignment is at the right bound
        values, deviation = alignments["right_alignment"], deviations["right_alignment"]
        if (max(values) - np.median(values)) / (max(values) - min(values) + 10e-6) <= 0.05:
            score = 1 - deviation / (0.02 * width)
        else:
            # Apply penalization
            score = 1 - 2 * deviation / (0.02 * width)

    return max(0.0, score)


def compute_columns_metrics(section: StructuredSection) -> tuple[float, float]:
    """
    Compute the column alignment metric for a structured section.
    :param section: The structured section to compute the metric for.
    :return: A tuple of the mean and minimum alignment score.
    """
    # Compute column alignment scores
    column_scores = [
        _col_alignment_score(col=col, char_length=section.char_length, width=section.width)
        for col in section.col_cells
    ]

    # Compute mean and minimum score
    valid_scores = [score for score in column_scores if score is not None]
    mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    min_score = min(valid_scores, default=0.0)
    return mean_score, min_score
