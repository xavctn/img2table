# coding: utf-8
from typing import List, Tuple

from img2table.tables.objects.cell import Cell


def get_column_delimiters(tb_clusters: List[List[Cell]], margin: int = 5) -> Tuple[List[int], List[List[Cell]]]:
    """
    Identify column delimiters from clusters
    :param tb_clusters: list of clusters composing the table
    :param margin: margin used for extremities
    :return: list of column delimiters and clusters corresponding to columns
    """
    # Compute horizontal bounds of each cluster
    bounds = [(min([c.x1 for c in cluster]), max([c.x2 for c in cluster])) for cluster in tb_clusters]

    # Group clusters that corresponds to the same column
    col_clusters = list()
    for i in range(len(bounds)):
        for j in range(i, len(bounds)):
            # If clusters overlap, put them in same column
            x_diff = min(bounds[i][1], bounds[j][1]) - max(bounds[i][0], bounds[j][0])
            overlap = x_diff / min(bounds[i][1] - bounds[i][0], bounds[j][1] - bounds[j][0]) >= 0.2
            if overlap:
                matching_clusters = [idx for idx, cl in enumerate(col_clusters) if {i, j}.intersection(cl)]
                if matching_clusters:
                    remaining_clusters = [cl for idx, cl in enumerate(col_clusters) if idx not in matching_clusters]
                    new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(col_clusters) if idx in matching_clusters])
                    col_clusters = remaining_clusters + [new_cluster]
                else:
                    col_clusters.append({i, j})

    # Compute column bounds
    col_bounds = sorted([(min([bounds[i][0] for i in col]), max([bounds[i][1] for i in col])) for col in col_clusters],
                        key=lambda x: sum(x))

    # Create delimiters
    x_delimiters = [round((left[1] + right[0]) / 2) for left, right in zip(col_bounds, col_bounds[1:])]
    x_delimiters = [col_bounds[0][0] - margin] + x_delimiters + [col_bounds[-1][1] + margin]

    return x_delimiters, [[c for idx in col for c in tb_clusters[idx]] for col in col_clusters]
