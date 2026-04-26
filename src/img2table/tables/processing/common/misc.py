from itertools import pairwise

import numpy as np


def _cluster_values(
    values: list[float] | list[int], median_gap_multiple: float, min_gap: float = 0.0
) -> list[int]:
    """
    Cluster values
    :param values: list of float values
    :param median_gap_multiple: multiple of median gap used as threshold for clustering
    :param min_gap: minimum gap enforced
    :return: cluster label (0, 1, 2, ...) for each value
    """
    if len(values) <= 1:
        return [0] * len(values)

    # Sort distinct values for gap computation, but keep cluster assignment for every input value
    sorted_values = sorted(set(values))

    # Compute gaps between consecutive sorted values
    gaps = [nxt - prv for prv, nxt in pairwise(sorted_values)]
    gap_threshold = median_gap_multiple * (
        np.median(gaps) if len(gaps) > 2 else min(gaps, default=0)
    )

    # Create clusters
    cluster_id, cluster_labels_sorted = 0, [0]
    for gap in gaps:
        if gap > max(gap_threshold, min_gap):
            cluster_id += 1
        cluster_labels_sorted.append(cluster_id)

    cluster_by_value = dict(zip(sorted_values, cluster_labels_sorted, strict=True))
    return [cluster_by_value[value] for value in values]
