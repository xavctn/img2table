# coding: utf-8
import statistics
from typing import List

from img2table.objects.tables import Cell
from img2table.utils.common import merge_contours


def merged_clusters(clusters: List[List[Cell]]) -> List[List[Cell]]:
    """
    From list of clusters, merge clusters that have common cells
    :param clusters: list of clusters
    :return: list of clusters wth disjoints cells
    """
    for i, v in enumerate(clusters):
        for j, k in enumerate(clusters[i + 1:], i + 1):
            if v & k:
                clusters[i] = v.union(clusters.pop(j))
                return merged_clusters(clusters)
    return clusters


def group_clusters(clusters: List[List[Cell]]) -> List[List[List[Cell]]]:
    """
    Create group of clusters that corresponds to a table
    :param clusters: list of cell clusters
    :return: groups of cell clusters that each corresponds to a table
    """
    # Loop over clusters to identify groups that can form a table
    cluster_groups = list()
    for idx, cluster in enumerate(clusters):
        if idx == 0:
            # Create first cluster group
            cluster_groups.append([sorted(cluster, key=lambda c: c.y1)])
        else:
            # Get vertical coordinates of the cluster
            y1_cluster = min([cell.y1 for cell in cluster])
            y2_cluster = max([cell.y2 for cell in cluster])
            matched = False
            # Loop over each existing cluster group to check if the cluster can be matched with it
            for i, cluster_group in enumerate(cluster_groups):
                # Get vertical coordinates of the cluster group
                y1_cg = min([cell.y1 for cl in cluster_group for cell in cl])
                y2_cg = max([cell.y2 for cl in cluster_group for cell in cl])

                # Overlapping y and determine if vertical coordinates corresponds
                overlapping_y = max(0, min(y2_cluster, y2_cg) - max(y1_cluster, y1_cg))
                y_corresponds = min(overlapping_y / (y2_cluster - y1_cluster), overlapping_y / (y2_cg - y1_cg)) >= 0.5

                # Compute average space between elements of cluster
                spacing_cl = [(cluster[idx + 1].y1 - cluster[idx].y1 + cluster[idx + 1].y2 - cluster[idx].y2) / 2
                              for idx in range(len(cluster) - 1)]
                avg_spacing_cl = statistics.mean(spacing_cl)

                # Compute average space between elements of the cluster group
                avg_spacing_cg = statistics.mean([(cl[idx + 1].y1 - cl[idx].y1 + cl[idx + 1].y2 - cl[idx].y2) / 2
                                                  for cl in cluster_group
                                                  for idx in range(len(cl) - 1)])

                # Compute if average spacing corresponds
                spacing_corresponds = abs(avg_spacing_cl / avg_spacing_cg - 1) <= 0.2

                # If spacing and height corresponds, append cluster to group
                if spacing_corresponds and y_corresponds:
                    cluster_groups[i].append(cluster)
                    matched = True

            # If the cluster has not been matched with any groups, create a new group from it
            if not matched:
                cluster_groups.append([cluster])

    # Reunite clusters that have cells in common
    dedup_cluster_groups = list()
    for cluster_group in cluster_groups:
        # For each group, check if some of the clusters contained in it intersect.
        # If it is the case, the intersecting clusters are merged into a larger one
        dedup_cluster_group = [merge_contours(contours=list(cl), vertically=True)
                               for cl in merged_clusters(clusters=list(map(set, cluster_group)))]

        # Append deduplicated group to the list
        dedup_cluster_groups.append([sorted(cl, key=lambda c: c.y1) for cl in dedup_cluster_group])

    # Filter on cluster tables that have at least 2 clusters
    dedup_cluster_groups = [cluster_group for cluster_group in dedup_cluster_groups if len(cluster_group) > 1]

    return dedup_cluster_groups
