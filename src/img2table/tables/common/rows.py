from __future__ import annotations

from collections import Counter
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from img2table.tables.common.misc import _cluster_values

if TYPE_CHECKING:
    from img2table.tables.borderless.types import MergedRow


def _evaluate_key_separation_value(
    rows: list[MergedRow],
    ref_separation: float,
    y_min: int,
    y_max: int,
    enforce_gap: bool = True,
) -> tuple[float, list]:
    """
    Evaluate pertinence of separation value based on created rows consistency
    :param rows: list of merged rows
    :param ref_separation: reference separation value
    :param y_min: minimum y value
    :param y_max: maximum y value
    :param enforce_gap: whether to enforce a gap between rows
    :return: key consistency score and created ranges
    """
    # Create sections
    sections = [[rows[0]]]
    for row in rows[1:]:
        prev_row = sections[-1][-1]
        separation = row.y_center - prev_row.y_center

        if enforce_gap and row.y1 < prev_row.y2:
            sections[-1].append(row)
        elif separation > 0.75 * ref_separation:
            sections.append([row])
        else:
            sections[-1].append(row)

    # Define ranges
    ranges = []
    current_y = y_min
    for prev_section, nxt_section in pairwise(sections):
        # Get middle point to compute separator between sections
        middle_point = int((prev_section[-1].y2 + nxt_section[0].y1) / 2)
        ranges.append((current_y, middle_point))
        current_y = middle_point

    # Add last range
    ranges.append((current_y, y_max))

    # Compute range heights
    range_heights = [rng[1] - rng[0] for rng in ranges]

    return 1 - np.std(range_heights) / (y_max - y_min) if len(ranges) > 1 else 0, ranges


def compute_row_ranges(
    rows: list[MergedRow], y_min: int, y_max: int, enforce_gap: bool = True
) -> list[tuple[int, int]]:
    """
    Identify vertical position ranges corresponding to rows
    :param rows: list of merged rows
    :param y_min: minimum y value
    :param y_max: maximum y value
    :param enforce_gap: whether to enforce a gap between rows
    :return: list of (start, end) vertical ranges
    """
    if len(rows) <= 1:
        return [(y_min, y_max)]

    # Compute separation between elements
    separations: list[float] = [nxt.y_center - prv.y_center for prv, nxt in pairwise(rows)]
    median_sep = np.median(separations)
    median_row_height = np.median([row.height for row in rows])

    # Cluster separations to identify distinct spacing patterns
    cluster_labels = _cluster_values(values=separations, median_gap_multiple=3)

    # Find the most common cluster that corresponds to rows to get eligible separations
    clusters_counter, eligible_separations = Counter(cluster_labels), {float(median_sep)}
    for cluster_id, cnt in clusters_counter.most_common(2):
        if cnt < max(clusters_counter.values()) / 5:
            # Not enough occurrences to be considered a distinct cluster
            continue

        cluster_separations = [
            sep
            for sep, label in zip(separations, cluster_labels, strict=True)
            if label == cluster_id
        ]
        if (cluster_median_sep := np.median(cluster_separations)) > median_row_height:
            eligible_separations.add(float(cluster_median_sep))

    # Evaluate best separation value
    best_score, best_ranges = 0, []
    for ref_sep in eligible_separations:
        score, ranges = _evaluate_key_separation_value(
            rows=rows, ref_separation=ref_sep, y_min=y_min, y_max=y_max, enforce_gap=enforce_gap
        )
        if score > best_score:
            best_score = score
            best_ranges = ranges

    return best_ranges or [(y_min, y_max)]
