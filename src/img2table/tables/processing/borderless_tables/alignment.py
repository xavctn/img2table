# coding : utf-8
from typing import List

import numpy as np

from img2table.tables.objects.cell import Cell


def left_aligned(cell_1: Cell, cell_2: Cell) -> bool:
    """
    Identify if two cells are left-aligned
    :param cell_1: Cell object
    :param cell_2: Cell object
    :return: boolean indicating if cells are left-aligned
    """
    # Compute minimum width of cells
    width = np.min([cell_1.width, cell_2.width])

    # Check for left alignment
    return abs(cell_1.x1 - cell_2.x1) / width < 0.1


def right_aligned(cell_1: Cell, cell_2: Cell) -> bool:
    """
    Identify if two cells are right-aligned
    :param cell_1: Cell object
    :param cell_2: Cell object
    :return: boolean indicating if cells are right-aligned
    """
    # Compute minimum width of cells
    width = np.min([cell_1.width, cell_2.width])

    # Check for right alignment
    return abs(cell_1.x2 - cell_2.x2) / width < 0.1


def center_aligned(cell_1: Cell, cell_2: Cell) -> bool:
    """
    Identify if two cells are center-aligned
    :param cell_1: Cell object
    :param cell_2: Cell object
    :return: boolean indicating if cells are center-aligned
    """
    # Compute minimum width of cells
    width = np.min([cell_1.width, cell_2.width])

    # Check for center alignment
    return abs(cell_1.x1 + cell_1.x2 - cell_2.x1 - cell_2.x2) / (2 * width) < 0.1


def aligned_cells(cell_1: Cell, cell_2: Cell) -> bool:
    """
    Identify if two cells are aligned vertically
    :param cell_1: Cell object
    :param cell_2: Cell object
    :return: boolean indicating if cells are aligned
    """
    # Check for width coherency
    if min(cell_1.width, cell_2.width) / max(cell_1.width, cell_2.width) < 0.25:
        return False

    # Check for each type of alignment
    return left_aligned(cell_1, cell_2) or right_aligned(cell_1, cell_2) or center_aligned(cell_1, cell_2)


def vertically_coherent_cluster(cluster: List[Cell]):
    """
    Verify if cluster if vertically coherent and split into sub-clusters if needed
    :param cluster: cluster of text contours based on alignment
    :return: vertically coherent clusters
    """
    # Sort cluster by vertical position
    cluster = sorted(cluster, key=lambda c: c.y1 + c.y2)

    # Compute median of distance between two elements of the cluster
    med_dist = np.median([bottom.y1 + bottom.y2 - top.y1 - top.y2 for top, bottom in zip(cluster, cluster[1:])])

    # Create sub clusters based on vertical distance
    seq = iter(cluster)
    subclusters = [[next(seq)]]
    for c in seq:
        distance = c.y1 + c.y2 - subclusters[-1][-1].y1 - subclusters[-1][-1].y2
        if distance > 2 * med_dist or distance < 0.5 * med_dist:
            subclusters.append([])
        subclusters[-1].append(c)

    return [subcl for subcl in subclusters if len(subcl) > 1]


def cluster_aligned_text(segment: List[Cell]) -> List[List[Cell]]:
    """
    Cluster text contours based on alignment
    :param segment: list of text contours as Cell objects
    :return: clusters of text contours based on alignment
    """
    # Sort cells in segments
    cells = sorted(set(segment), key=lambda c: (c.y1, c.x1))

    # Create clusters based on alignment of cells
    total_clusters = list()
    for func in (left_aligned, right_aligned, center_aligned):
        clusters = list()
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                aligned = func(cells[i], cells[j])
                # If cells are adjacent, find matching clusters
                if aligned:
                    matching_clusters = [idx for idx, cl in enumerate(clusters) if {i, j}.intersection(cl)]
                    if matching_clusters:
                        remaining_clusters = [cl for idx, cl in enumerate(clusters) if idx not in matching_clusters]
                        new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(clusters) if idx in matching_clusters])
                        clusters = remaining_clusters + [new_cluster]
                    else:
                        clusters.append({i, j})

        total_clusters.extend(clusters)

    # Check if there are some clusters
    if len(total_clusters) == 0:
        return []

    # Get maximal clusters
    seq = iter(sorted(total_clusters, key=lambda cc: len(cc), reverse=True))
    dedup_clusters = [next(seq)]
    for cl in seq:
        if not any([cl.intersection(c) == cl for c in dedup_clusters]):
            dedup_clusters.append(cl)

    # Get cells in clusters
    clusters = [[cells[idx] for idx in cl] for cl in dedup_clusters]

    # Get vertically coherent clusters
    v_clusters = [subcl for cluster in clusters for subcl in vertically_coherent_cluster(cluster=cluster)]

    # Order clusters by horizontal position
    v_clusters = sorted(v_clusters, key=lambda cl: np.mean([c.x1 + c.x2 for c in cl]))

    return v_clusters
