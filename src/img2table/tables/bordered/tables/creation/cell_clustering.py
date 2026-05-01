from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from img2table.tables.common import find_components

if TYPE_CHECKING:
    from img2table.tables.types import Cell


def get_adjacent_cells(cells: list[Cell]) -> list[set[int]]:
    """
    Identify adjacent cells
    :param cells: list of cells
    :return: list of sets of adjacent cells indexes
    """
    if len(cells) == 0:
        return []

    n = len(cells)
    x1 = np.array([c.x1 for c in cells], dtype=np.float32)
    y1 = np.array([c.y1 for c in cells], dtype=np.float32)
    x2 = np.array([c.x2 for c in cells], dtype=np.float32)
    y2 = np.array([c.y2 for c in cells], dtype=np.float32)
    widths = x2 - x1
    heights = y2 - y1

    # Compute overlaps
    x_overlap = np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :])
    y_overlap = np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :])

    # Compute distances
    diff_x = np.minimum(
        np.minimum(np.abs(x1[:, None] - x1[None, :]), np.abs(x1[:, None] - x2[None, :])),
        np.minimum(np.abs(x2[:, None] - x1[None, :]), np.abs(x2[:, None] - x2[None, :])),
    )
    diff_y = np.minimum(
        np.minimum(np.abs(y1[:, None] - y1[None, :]), np.abs(y1[:, None] - y2[None, :])),
        np.minimum(np.abs(y2[:, None] - y1[None, :]), np.abs(y2[:, None] - y2[None, :])),
    )

    # Thresholds
    thresh_x = np.minimum(5.0, 0.05 * np.minimum(widths[:, None], widths[None, :]))
    thresh_y = np.minimum(5.0, 0.05 * np.minimum(heights[:, None], heights[None, :]))

    # Adjacency: upper triangle only (i < j) avoids duplicate pairs
    adjacent = (y_overlap > 5) & (diff_x <= thresh_x) | (x_overlap > 5) & (diff_y <= thresh_y)
    rows_idx, cols_idx = np.where(np.triu(adjacent, k=1))
    result: list[set[int]] = [{int(i), int(j)} for i, j in zip(rows_idx, cols_idx, strict=True)]

    # Add cells not appearing in any pair
    result.extend({i} for i in range(n) if i not in set().union(*result))

    return result


def cluster_cells_in_tables(cells: list[Cell]) -> list[list[Cell]]:
    """
    Based on adjacent cells, create clusters of cells that corresponds to tables
    :param cells: list cells in image
    :return: list of list of cells, representing several clusters of cells that form a table
    """
    # Get couples of adjacent cells
    adjacent_cells = get_adjacent_cells(cells=cells)

    # Loop over couples to create clusters
    clusters = find_components(edges=adjacent_cells)

    # Return list of cell objects
    return [[cells[idx] for idx in cl] for cl in clusters]
