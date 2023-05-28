# coding: utf-8
from typing import List, Any, Callable


def cluster_items(items: List[Any], clustering_func: Callable) -> List[List[Any]]:
    """
    Cluster items based on a function
    :param items: list of items
    :param clustering_func: clustering function
    :return: list of list of items based on clustering function
    """
    # Create clusters based on clustering function between items
    clusters = list()
    for i in range(len(items)):
        for j in range(i, len(items)):
            # Check if both items corresponds according to the clustering function
            corresponds = clustering_func(items[i], items[j])

            # If both items correspond, find matching clusters or create a new one
            if corresponds:
                matching_clusters = [idx for idx, cl in enumerate(clusters) if {i, j}.intersection(cl)]
                if matching_clusters:
                    remaining_clusters = [cl for idx, cl in enumerate(clusters) if idx not in matching_clusters]
                    new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(clusters) if idx in matching_clusters])
                    clusters = remaining_clusters + [new_cluster]
                else:
                    clusters.append({i, j})

    return [[items[idx] for idx in c] for c in clusters]
