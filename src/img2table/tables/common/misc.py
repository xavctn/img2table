from collections import defaultdict
from collections.abc import Callable, Collection, Hashable, Sequence
from itertools import pairwise
from typing import TypeVar

import numpy as np

T = TypeVar("T", bound=Hashable)


def find_components(edges: Sequence[Collection[T]]) -> list[list[T]]:
    # Construct adjacency mapping
    adjacency_mapping = defaultdict(set)
    for edge in map(list, edges):
        for cmp in edge:
            adjacency_mapping[cmp].add(cmp)
        if len(edge) == 2:
            cmp1, cmp2 = edge
            adjacency_mapping[cmp1].add(cmp2)
            adjacency_mapping[cmp2].add(cmp1)

    # DFS
    visited, components = set(), []
    for node in adjacency_mapping:
        if node in visited:
            continue

        stack, component = [node], []
        visited.add(node)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency_mapping[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def cluster_items(items: list[T], clustering_func: Callable[[T, T], bool]) -> list[list[T]]:
    """
    Cluster items based on a function
    :param items: list of items
    :param clustering_func: clustering function
    :return: list of list of items based on clustering function
    """
    # Create clusters based on clustering function between items
    edges: list[set[int]] = []
    for i in range(len(items)):
        edges.append({i})
        for j in range(i + 1, len(items)):
            # Check if both items corresponds according to the clustering function
            corresponds = clustering_func(items[i], items[j])

            # If both items correspond, find matching clusters or create a new one
            if corresponds:
                edges.append({i, j})

    return [[items[idx] for idx in c] for c in find_components(edges=edges)]


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
