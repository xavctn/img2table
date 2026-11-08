from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

    from img2table.tables.types import Line, Table


def is_graphic_region(
    thresh: np.ndarray, table: Table, char_length: float, lines: list[Line]
) -> bool:
    """
    Reject small borderless candidates dominated by graphic components.
    :param thresh: original threshold image before line removal
    :param table: proposed borderless table
    :param char_length: typical character size
    :param lines: detected graphical lines
    :return: whether the candidate is more likely to be a graphic
    """
    if table.nb_rows > 6:
        return False

    crop = thresh[table.y1 : table.y2, table.x1 : table.x2]
    _, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    components = stats[1:]
    ink = components[:, cv2.CC_STAT_AREA].sum()
    if ink == 0:
        return False

    text_components = components[
        (components[:, cv2.CC_STAT_WIDTH] >= 0.5 * char_length)
        & (components[:, cv2.CC_STAT_WIDTH] <= 2.5 * char_length)
        & (components[:, cv2.CC_STAT_HEIGHT] >= char_length)
        & (components[:, cv2.CC_STAT_HEIGHT] <= 2.5 * char_length)
        & (components[:, cv2.CC_STAT_AREA] >= 0.15 * char_length**2)
    ]
    text_ink_ratio = text_components[:, cv2.CC_STAT_AREA].sum() / ink
    if text_ink_ratio < 0.1:
        return True
    if text_ink_ratio >= 0.4:
        return False

    return any(
        line.vertical
        and table.x1 <= line.x1 <= table.x2
        and line.y1 < table.y2
        and line.y2 > table.y1
        and (table.y1 - line.y1 > table.height or line.y2 - table.y2 > table.height)
        for line in lines
    )
