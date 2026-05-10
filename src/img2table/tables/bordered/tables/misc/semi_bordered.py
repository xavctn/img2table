from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from img2table.tables.bordered.tables.normalization import normalize_table_cells
from img2table.tables.types import Cell

if TYPE_CHECKING:
    from img2table.tables.types import Line


@dataclass
class ClusterCharacteristics:
    x_values: list[int]
    y_values: list[int]
    char_length: float

    @property
    def x_min(self) -> int:
        return min(self.x_values)

    @property
    def x_max(self) -> int:
        return max(self.x_values)

    @property
    def y_min(self) -> int:
        return min(self.y_values)

    @property
    def y_max(self) -> int:
        return max(self.y_values)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def h_tolerance(self) -> float:
        return max(2 * self.char_length, 0.05 * self.width)

    @property
    def v_tolerance(self) -> float:
        return max(2 * self.char_length, 0.05 * self.height)


def get_cluster_characteristics(cluster: list[Cell], char_length: float) -> ClusterCharacteristics:
    """
    Compute cluster characteristics
    :param cluster: list of cells in cluster
    :param char_length: average character length
    :return: cluster characteristics
    """
    x_values = sorted({c.x1 for c in cluster} | {c.x2 for c in cluster})
    y_values = sorted({c.y1 for c in cluster} | {c.y2 for c in cluster})

    return ClusterCharacteristics(x_values=x_values, y_values=y_values, char_length=char_length)


def get_lines_in_cluster(
    lines: list[Line], characteristics: ClusterCharacteristics
) -> tuple[list[Line], list[Line]]:
    """
    Identify list of lines belonging to cluster
    :param lines: list of lines in image
    :param characteristics: cluster characteristics
    :return: list of horizontal and vertical lines in cluster
    """
    # Keep horizontal lines aligned with cluster rows and overlapping the table span
    h_lines_cl = sorted(
        [
            ln
            for ln in lines
            if ln.horizontal
            and min(abs(ln.y1 - y_val) for y_val in characteristics.y_values)
            <= characteristics.v_tolerance
            and min(ln.x2, characteristics.x_max) - max(ln.x1, characteristics.x_min)
            >= 0.5 * min(ln.x2 - ln.x1, characteristics.x_max - characteristics.x_min)
        ],
        key=lambda ln: ln.y1,
    )

    # Keep vertical lines aligned with cluster columns and overlapping the table span
    v_lines_cl = sorted(
        [
            ln
            for ln in lines
            if ln.vertical
            and min(abs(ln.x1 - x_val) for x_val in characteristics.x_values)
            <= characteristics.h_tolerance
            and min(ln.y2, characteristics.y_max) - max(ln.y1, characteristics.y_min)
            >= 0.5 * min(ln.y2 - ln.y1, characteristics.y_max - characteristics.y_min)
        ],
        key=lambda ln: ln.x1,
    )

    return h_lines_cl, v_lines_cl


def identify_table_dimensions(
    characteristics: ClusterCharacteristics,
    h_lines_cl: list[Line],
    v_lines_cl: list[Line],
) -> tuple[int, int, int, int]:
    """
    Identify table dimensions by checking lines corresponding to cluster
    :param characteristics: cluster characteristics
    :param h_lines_cl: list of horizontal lines in cluster
    :param v_lines_cl: list of vertical lines in cluster
    :return: tuple of cluster dimensions
    """
    # Identify left and right bounds
    if h_lines_cl:
        top_line, bottom_line = h_lines_cl[0], h_lines_cl[-1]

        # Left bound
        left_bounds = {top_line.x1, bottom_line.x1}
        left = (
            min(left_bounds)
            if max(left_bounds) - min(left_bounds) <= characteristics.h_tolerance
            else characteristics.x_min
        )

        # Right bound
        right_bounds = {top_line.x2, bottom_line.x2}
        right = (
            min(right_bounds)
            if max(right_bounds) - min(right_bounds) <= characteristics.h_tolerance
            else characteristics.x_max
        )
    else:
        left, right = characteristics.x_min, characteristics.x_max

    # Identify top and bottom bounds
    if v_lines_cl:
        left_line, right_line = v_lines_cl[0], v_lines_cl[-1]

        # Top bound
        top_bounds = {left_line.y1, right_line.y1}
        top = (
            min(top_bounds)
            if max(top_bounds) - min(top_bounds) <= characteristics.v_tolerance
            else characteristics.y_min
        )

        # Bottom bound
        bottom_bounds = {left_line.y2, right_line.y2}
        bottom = (
            min(bottom_bounds)
            if max(bottom_bounds) - min(bottom_bounds) <= characteristics.v_tolerance
            else characteristics.y_max
        )
    else:
        top, bottom = characteristics.y_min, characteristics.y_max

    return left, right, top, bottom


def has_line_support(
    candidate: Cell,
    h_lines_cl: list[Line],
    v_lines_cl: list[Line],
    tolerance: float,
) -> bool:
    """
    Check whether a candidate cell has line support
    :param candidate: candidate cell
    :param h_lines_cl: list of horizontal lines in cluster
    :param v_lines_cl: list of vertical lines in cluster
    :param tolerance: border alignment tolerance
    :return: boolean indicating if the candidate should be kept
    """
    # Combine horizontal line support
    top_supported = any(
        ln
        for ln in h_lines_cl
        if abs(ln.y1 - candidate.y1) <= tolerance
        and min(ln.x2, candidate.x2) - max(ln.x1, candidate.x1) >= 0.9 * candidate.width
    )
    bottom_supported = any(
        ln
        for ln in h_lines_cl
        if abs(ln.y1 - candidate.y2) <= tolerance
        and min(ln.x2, candidate.x2) - max(ln.x1, candidate.x1) >= 0.9 * candidate.width
    )
    horizontal_support = top_supported or bottom_supported

    # Combine vertical line support
    left_supported = any(
        ln
        for ln in v_lines_cl
        if abs(ln.x1 - candidate.x1) <= tolerance
        and min(ln.y2, candidate.y2) - max(ln.y1, candidate.y1) >= 0.9 * candidate.height
    )
    right_supported = any(
        ln
        for ln in v_lines_cl
        if abs(ln.x1 - candidate.x2) <= tolerance
        and min(ln.y2, candidate.y2) - max(ln.y1, candidate.y1) >= 0.9 * candidate.height
    )
    vertical_support = left_supported or right_supported

    return horizontal_support and vertical_support


def identify_potential_new_cells(
    cluster: list[Cell],
    h_lines_cl: list[Line],
    v_lines_cl: list[Line],
    left: int,
    right: int,
    top: int,
    bottom: int,
    tolerance: float = 2.0,
) -> list[Cell]:
    """
    Indetify potential new cells in cluster based on computed dimensions
    :param cluster: cluster of cells
    :param h_lines_cl: list of horizontal lines in cluster
    :param v_lines_cl: list of vertical lines in cluster
    :param left: left coordinate of cluster
    :param right: right coordinate of cluster
    :param top: top coordinate of cluster
    :param bottom: bottom coordinate of cluster
    :param tolerance: tolerance for distances
    :return: list of potential cells
    """
    # Cells array
    cells_arr = np.array([[c.x1, c.y1, c.x2, c.y2, c.area] for c in cluster])

    # Build the candidate grid from current delimiters and supported outer bounds
    x_cluster = sorted({c.x1 for c in cluster} | {c.x2 for c in cluster} | {left, right})
    y_cluster = sorted({c.y1 for c in cluster} | {c.y2 for c in cluster} | {top, bottom})

    # Keep only supported empty rectangles from the reconstructed grid
    existing_cells = set(cluster)
    new_cells = []
    for x1, x2 in pairwise(x_cluster):
        for y1, y2 in pairwise(y_cluster):
            candidate = Cell(x1=x1, y1=y1, x2=x2, y2=y2)
            if candidate.area == 0 or candidate in existing_cells:
                continue

            # Check overlap with existing cells
            overlaps = np.multiply(
                np.maximum(
                    0,
                    np.minimum(cells_arr[:, 2], candidate.x2)
                    - np.maximum(cells_arr[:, 0], candidate.x1),
                ),
                np.maximum(
                    0,
                    np.minimum(cells_arr[:, 3], candidate.y2)
                    - np.maximum(cells_arr[:, 1], candidate.y1),
                ),
            )
            if np.sum(overlaps >= 0.2 * np.minimum(cells_arr[:, 4], candidate.area)) > 0:
                continue

            if not has_line_support(
                candidate=candidate,
                h_lines_cl=h_lines_cl,
                v_lines_cl=v_lines_cl,
                tolerance=tolerance,
            ):
                continue
            new_cells.append(candidate)

    return new_cells


def update_cluster_cells(
    cluster: list[Cell], new_cells: list[Cell], char_length: float
) -> list[Cell]:
    """
    Update cluster cells with new ones if relevant
    :param cluster: cluster of cells
    :param new_cells: list of potential new cells
    :param char_length: average character length
    :return: list of updated cluster cells
    """
    if len(new_cells) == 0:
        return cluster

    # Deduplicate synthesized cells before normalizing the final grid
    cluster_set = set(cluster)
    final_cells = [cell for cell in dict.fromkeys(new_cells) if cell not in cluster_set]

    if final_cells:
        return normalize_table_cells(cluster_cells=cluster + final_cells, char_length=char_length)
    return cluster


def add_semi_bordered_cells(
    cluster: list[Cell], lines: list[Line], char_length: float
) -> list[Cell]:
    """
    Identify and add semi-bordered cells to cluster
    :param cluster: cluster of cells
    :param lines: lines in image
    :param char_length: average character length
    :return: cluster with add semi-bordered cells
    """
    if len(cluster) == 0:
        return cluster

    # Compute cluster characteristics
    characteristics = get_cluster_characteristics(cluster=cluster, char_length=char_length)

    # Identify lines used in cluster
    h_lines_cl, v_lines_cl = get_lines_in_cluster(lines=lines, characteristics=characteristics)

    # Get dimensions of new cluster
    left, right, top, bottom = identify_table_dimensions(
        characteristics=characteristics,
        h_lines_cl=h_lines_cl,
        v_lines_cl=v_lines_cl,
    )

    # Identify potential new cells
    new_cells = identify_potential_new_cells(
        cluster=cluster,
        h_lines_cl=h_lines_cl,
        v_lines_cl=v_lines_cl,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )

    return update_cluster_cells(cluster=cluster, new_cells=new_cells, char_length=char_length)
