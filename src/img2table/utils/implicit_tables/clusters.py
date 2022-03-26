# coding: utf-8
from typing import List

from img2table.objects.tables import Cell, Table


def is_contained_table(cell: Cell, table: Table) -> bool:
    """
    Determine if a cell is contained in a table
    :param cell: Cell object
    :param table: Table object
    :return: boolean indicating if the cell is contained in the table
    """
    # Compute common coordinates
    x_left = max(cell.x1, table.x1)
    y_top = max(cell.y1, table.y1)
    x_right = min(cell.x2, table.x2)
    y_bottom = min(cell.y2, table.y2)

    if x_right < x_left or y_bottom < y_top:
        return False

    # Compute intersection area as well as cell area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    cell_area = cell.height * cell.width

    # Compute percentage of intersection with cell area
    iou = intersection_area / cell_area

    return iou >= 0.9


def make_coherent_cluster(cluster: List[Cell]) -> List[List[Cell]]:
    """
    Split cluster into vertically coherent sub clusters
    :param cluster: cluster of cells
    :return: list of vertically coherent sub clusters
    """
    # Sort the cluster vertically
    sorted_cluster = sorted(cluster, key=lambda c: c.y1)

    # Loop over cells in the cluster and check vertical coherency
    clusters = list()
    for idx, cell in enumerate(sorted_cluster):
        if idx == 0:
            # Set cluster, vertical difference between cells, average cell height and width
            cluster = [cell]
            avg_vertical_diff = None
            avg_cell_height = cell.y2 - cell.y1
            avg_cell_length = cell.x2 - cell.x1
        else:
            # Compute vertical difference between the cell and the last element of the cluster, cell height and length
            v_diff = (cell.y2 + cell.y1 - cluster[-1].y2 - cluster[-1].y1) / 2
            cell_height = cell.y2 - cell.y1
            cell_length = cell.x2 - cell.x1
            # Check if vertical difference is coherent with cluster
            if avg_vertical_diff is None or 0.5 * avg_vertical_diff < v_diff < 2 * avg_vertical_diff:
                # Check if cell length is coherent with cluster
                if 0.2 * avg_cell_length <= cell_length <= 5 * avg_cell_length:
                    avg_cell_height = (avg_cell_height * len(cluster) + cell_height) / (len(cluster) + 1)
                    avg_cell_length = (avg_cell_length * len(cluster) + cell_length) / (len(cluster) + 1)
                    avg_vertical_diff = ((avg_vertical_diff or v_diff) * (len(cluster) - 1) + v_diff) / len(cluster)
                    # Check if vertical difference is coherent with cell height
                    if avg_vertical_diff / avg_cell_height <= 3:
                        cluster.append(cell)
                        continue

            # If the vertical difference is not coherent with the cluster, post the cluster and open a new one
            if len(cluster) > 1:
                clusters.append(cluster)
            cluster = [cell]
            vertical_diff = None
            avg_cell_height = cell.y2 - cell.y1
            avg_cell_length = cell.x2 - cell.x1

    if len(cluster) > 1:
        clusters.append(cluster)

    return clusters


def cluster_contours(contours: List[Cell], tables: List[Table], max_cluster_diff: int = 5) -> List[List[Cell]]:
    """
    Create clusters of contours that are horizontally aligned and vertically coherent
    :param contours: list of contours of the image
    :param tables: list of existing Table object in image
    :param max_cluster_diff: maximum horizontal difference to form a cluster
    :return: list of clusters of contours that are horizontally aligned and vertically coherent
    """
    # Identify coherent tables (i.e tables that have at least multiple columns or rows)
    coherent_tables = [table for table in tables if table.nb_columns * table.nb_rows > 1]

    # Identify clusters of left aligned, centered aligned and right aligned contours
    left_aligned_clusters = list()
    centered_aligned_clusters = list()
    right_aligned_clusters = list()

    for cell in contours:
        # If cell is already contained in a table, do not use it
        if coherent_tables:
            if max([is_contained_table(cell=cell, table=table) for table in coherent_tables]):
                continue

        # Identify left aligned clusters
        _left_aligned = [idx for idx, cluster in enumerate(left_aligned_clusters)
                         if abs(cell.x1 - cluster[0].x1) <= max_cluster_diff]
        if _left_aligned:
            for idx in _left_aligned:
                left_aligned_clusters[idx] += [cell]
        else:
            left_aligned_clusters.append([cell])

        # Identify centered aligned clusters
        _center_aligned = [idx for idx, cluster in enumerate(centered_aligned_clusters)
                           if abs((cell.x2 + cell.x1) / 2 - (cluster[0].x2 + cluster[0].x1) / 2) <= max_cluster_diff]
        if _center_aligned:
            for idx in _center_aligned:
                centered_aligned_clusters[idx] += [cell]
        else:
            centered_aligned_clusters.append([cell])

        # Identify right aligned clusters
        _right_aligned = [idx for idx, cluster in enumerate(right_aligned_clusters)
                          if abs(cell.x2 - cluster[0].x2) <= max_cluster_diff]
        if _right_aligned:
            for idx in _right_aligned:
                right_aligned_clusters[idx] += [cell]
        else:
            right_aligned_clusters.append([cell])

    # Keep only clusters with at least 2 iterations
    left_aligned_clusters = [cluster for cluster in left_aligned_clusters if len(cluster) > 1]
    centered_aligned_clusters = [cluster for cluster in centered_aligned_clusters if len(cluster) > 1]
    right_aligned_clusters = [cluster for cluster in right_aligned_clusters if len(cluster) > 1]

    # Merge all clusters
    clusters = list()
    for idx, cluster in enumerate(sorted(left_aligned_clusters + centered_aligned_clusters + right_aligned_clusters,
                                         key=lambda x: len(x),
                                         reverse=True)):
        if idx == 0:
            clusters.append(cluster)
        elif min([len(set(cluster) - set(cl)) for cl in clusters]) > 0:
            clusters.append(cluster)

    # Split clusters into vertically coherent clusters
    return [v_cluster for cluster in clusters for v_cluster in make_coherent_cluster(cluster=cluster)]
