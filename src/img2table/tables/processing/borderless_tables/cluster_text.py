# coding : utf-8
from typing import List

import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.common import merge_contours


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
    return abs(cell_1.x1 + cell_1.x2 - cell_2.x1 - cell_2.x2) / 2 * width < 0.1


def aligned_cells(cell_1: Cell, cell_2: Cell) -> bool:
    """
    Identify if two cells are aligned vertically
    :param cell_1: Cell object
    :param cell_2: Cell object
    :return: boolean indicating if cells are aligned
    """
    # Check for each type of alignment
    return left_aligned(cell_1, cell_2) or right_aligned(cell_1, cell_2) or center_aligned(cell_1, cell_2)


def cluster_text_contours(segment: List[Cell]) -> List[List[Cell]]:
    """
    Cluster text contours based on alignment
    :param segment: list of text contours as Cell objects
    :return: clusters of text contours based on alignment
    """
    # Sort cells in segments
    cells = sorted(set(segment), key=lambda c: (c.y1, c.x1))

    # Create clusters based on alignment of cells
    clusters = list()
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            aligned = aligned_cells(cells[i], cells[j])
            # If cells are adjacent, find matching clusters
            if aligned:
                matching_clusters = [idx for idx, cl in enumerate(clusters) if {i, j}.intersection(cl)]
                if matching_clusters:
                    remaining_clusters = [cl for idx, cl in enumerate(clusters) if idx not in matching_clusters]
                    new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(clusters) if idx in matching_clusters])
                    clusters = remaining_clusters + [new_cluster]
                else:
                    clusters.append({i, j})

    # Merge corresponding cells in each cluster
    clusters = [merge_contours(contours=[cells[idx] for idx in cl]) for cl in clusters]

    # Order clusters by horizontal position
    clusters = sorted(clusters, key=lambda cl: np.mean([c.x1 + c.x2 for c in cl]))

    return clusters
