from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from img2table.tables.borderless.tables.filter.model import (
        StructuredSection,
    )


def content_spacing_consistency(section: StructuredSection) -> float:
    """
    Check if vertical spacing between content rows is consistent
    :param section: The structured section to compute the metric for.
    :return: score between 0 and 1
    """
    if not section.items:
        return 0.0

    range_rows, range_heights = [], []
    for y_start, y_end in section.row_ranges():
        range_heights.append(y_end - y_start)
        range_rows.append(
            [
                row
                for row in section.merged_rows
                if min(row.y2, y_end) - max(row.y1, y_start) >= 0.5 * row.height
            ]
        )

    # Compute separations between consecutive rows
    range_seps: list[float] = []
    for prv_rows, nxt_rows in pairwise(range_rows):
        if not prv_rows or not nxt_rows:
            continue
        prv_y = max(row.y_center for row in prv_rows)
        nxt_y = min(row.y_center for row in nxt_rows)
        range_seps.append(nxt_y - prv_y)

    if not range_seps:
        return 0.0

    # Apply penalization
    spacing_regularization = max(0.025 * section.height, 3 * section.char_length)
    return max(
        0, float(1 - max(np.std(range_seps), np.std(range_heights)) / spacing_regularization)
    )
