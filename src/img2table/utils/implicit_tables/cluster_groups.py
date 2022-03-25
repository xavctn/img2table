# coding: utf-8
import statistics
from typing import List

from img2table.objects.tables import Cell


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

                # Find if y1 and y2 corresponds
                y1_corresponds = abs(y1_cluster - y1_cg) / (y2_cg - y1_cg) <= 0.05
                y2_corresponds = abs(y2_cluster - y2_cg) / (y2_cg - y1_cg) <= 0.05

                # If corresponds, append to cluster
                if y1_corresponds and y2_corresponds:
                    cluster_groups[i].append(cluster)
                    matched = True
                    continue

                # Compute average space between elements of cluster
                spacing_cl = [(cluster[idx + 1].y1 - cluster[idx].y1 + cluster[idx + 1].y2 - cluster[idx].y2) / 2
                              for idx in range(len(cluster) - 1)]
                avg_spacing_cl = statistics.mean(spacing_cl)

                # Compute average space between elements of the cluster group
                avg_spacing_cg = statistics.mean([(cl[idx + 1].y1 - cl[idx].y1 + cl[idx + 1].y2 - cl[idx].y2) / 2
                                                  for cl in cluster_group
                                                  for idx in range(len(cl) - 1)])

                # Compute if average spacing and height corresponds
                spacing_corresponds = abs(avg_spacing_cl / avg_spacing_cg - 1) <= 0.2
                height_corresponds = abs((y2_cluster - y1_cluster) / (y2_cg - y1_cg) - 1) <= 0.5

                # If one the vertical ends, spacing anf height corresponds, append cluster to group
                if (y1_corresponds or y2_corresponds) and spacing_corresponds and height_corresponds:
                    cluster_groups[i].append(cluster)
                    matched = True
                    continue

            # If the cluster has not been matched with any groups, create a new group from it
            if not matched:
                cluster_groups.append([cluster])

    # Reunite clusters that have cells in common
    dedup_cluster_groups = list()
    for cluster_group in cluster_groups:
        # For each group, check if some of the clusters contained in it intersect.
        # If it is the case, the intersecting clusters are merged into a larger one
        dedup_cluster_group = list()
        _idx = list(range(len(cluster_group)))
        for idx in iter(_idx):
            # Check if some clusters intersect the current one
            common_clusters = [i for i, cluster in enumerate(cluster_group)
                               if len(set(cluster_group[idx]).intersection(cluster_group[i])) > 0
                               and i > idx]
            if common_clusters:
                # Create a new cluster by merging all intersecting clusters
                _cluster = list(set([cell for j, cluster in enumerate(cluster_group) for cell in cluster
                                     if j in [idx] + common_clusters]))
                dedup_cluster_group.append(sorted(_cluster, key=lambda c: c.y1))
                # Remove merged clusters from the list
                for j in common_clusters:
                    _idx.remove(j)
            else:
                dedup_cluster_group.append(cluster_group[idx])

        # Append deduplicated group to the list
        dedup_cluster_groups.append(dedup_cluster_group)

    # Filter on cluster tables that have at least 2 clusters
    dedup_cluster_groups = [cluster_group for cluster_group in dedup_cluster_groups if len(cluster_group) > 1]

    return dedup_cluster_groups
