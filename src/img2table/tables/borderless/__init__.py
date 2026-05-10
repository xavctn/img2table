from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.tables.borderless.layout import identify_image_layout
from img2table.tables.borderless.sections import identify_column_sections
from img2table.tables.borderless.tables import identify_tables

if TYPE_CHECKING:
    import numpy as np

    from img2table.tables.types import Line, Table


def extract_borderless_tables(
    thresh: np.ndarray,
    lines: list[Line],
    char_length: float,
    existing_tables: list[Table] | None = None,
) -> list[Table]:
    """
    Identify borderless tables in a thresholded image.
    :param thresh: Thresholded image.
    :param lines: List of detected lines.
    :param char_length: Character length in pixels.
    :param existing_tables: Existing bordered tables.
    """
    h, w = thresh.shape[:2]

    # Identify layout from the thresholded image
    layout_regions = identify_image_layout(
        thresh=thresh, lines=lines, char_length=char_length, existing_tables=existing_tables
    )

    tables = []
    for region in layout_regions:
        # Get column sections
        column_sections = identify_column_sections(layout_region=region, min_width=char_length)

        # Identify tables
        tables += identify_tables(
            column_sections=column_sections,
            char_length=char_length,
            height=h,
            width=w,
        )

    # Filter out tables that overlap with existing tables
    return [
        tb
        for tb in tables
        if not any(tb.overlaps(existing_tb, pct=0.2) for existing_tb in existing_tables or [])
    ]
