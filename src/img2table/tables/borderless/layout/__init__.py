from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.tables.borderless.layout.layout import identify_layout
from img2table.tables.borderless.layout.text_lines import (
    identify_image_contours,
)

if TYPE_CHECKING:
    import numpy as np

    from img2table.tables.borderless.types import LayoutRegion
    from img2table.tables.types import Line, Table


def identify_image_layout(
    thresh: np.ndarray,
    lines: list[Line],
    char_length: float,
    existing_tables: list[Table] | None = None,
) -> list[LayoutRegion]:
    """
    Identify layout regions in image
    :param thresh: threshold image array
    :param lines: list of image rows
    :param char_length: average character length
    :param existing_tables: list of detected bordered tables
    :return: list of layout regions
    """
    # Identify contours in the thresholded image
    contours = identify_image_contours(
        thresh=thresh, lines=lines, char_length=char_length, existing_tables=existing_tables
    )

    # Get layout regions
    return identify_layout(
        contours=contours, lines=lines, min_width=2 * char_length, width=thresh.shape[1]
    )
