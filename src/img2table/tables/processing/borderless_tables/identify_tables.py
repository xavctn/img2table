# coding: utf-8
from itertools import combinations
from typing import Dict, List, Set, Iterator

from img2table.tables.objects.cell import Cell


def find_common_rows(cluster_1: List[Cell], cluster_2: List[Cell]) -> int:
    """
    Count number of vertically coherent cells between two clusters
    :param cluster_1: cluster as list of Cell objects
    :param cluster_2: cluster as list of Cell objects
    :return: number of common rows between the two clusters
    """
    # Compute cluster height
    nb_rows = max(len(cluster_1), len(cluster_2))
    height = max([c.y2 for c in cluster_1 + cluster_2]) - min([c.y1 for c in cluster_1 + cluster_2])

    # Loop over cells to get corresponding ones
    c1_cells, c2_cells = set(), set()
    for cell in cluster_1:
        # Get corresponding cells in cluster 2
        corresponding_cells = [(cell, c) for c in cluster_2
                               if abs(c.y1 + c.y2 - cell.y1 - cell.y2) / (2 * height) <= 1 / (4 * nb_rows)]
        c1_cells = c1_cells.union(set([link[0] for link in corresponding_cells]))
        c2_cells = c2_cells.union(set([link[1] for link in corresponding_cells]))

    return min(map(len, [c1_cells, c2_cells]))


def dfs(graph: Dict[int, List[int]], start: int , end: int) -> Iterator[List[int]]:
    """
    Depth-first search algorithm in directed graph
    :param graph: dict representing graph
    :param start: starting node
    :param end: ending node
    :return: paths between starting and ending node
    """
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path + [next_state]))


def get_maximal_cycles(cycles: List[Set]) -> List[Set]:
    """
    From list of sets, keep only maximal ones
    :param cycles: list of sets
    :return: list of maximal sets
    """
    if cycles:
        # Get maximal cycles
        seq = iter(sorted(cycles, key=lambda cc: len(cc), reverse=True))
        max_cycles = [next(seq)]
        for cycle in seq:
            if not any([cycle.intersection(c) == cycle for c in max_cycles]):
                max_cycles.append(cycle)

        return max_cycles
    return []


def match_with_cycle(cluster: List[Cell], cycle: List[List[Cell]]) -> bool:
    """
    Assert if a cluster can be matched with a cycle
    :param cluster: cluster of cells
    :param cycle: list of cell clusters
    :return: boolean indicating if the cluster can be matched with a cycle
    """
    # Sort cluster
    cluster = sorted(cluster, key=lambda c: c.y1 + c.y2)

    # Compute cycle height
    nb_rows = max(map(len, cycle))
    height = max(map(lambda cl: max([c.y2 for c in cl]) - min([c.y1 for c in cl]), cycle))

    # Get all cells in cycle
    cycle_cells = [c for cl in cycle for c in cl]

    matching_cells = list()
    # Try matching cluster cells with cycle cells
    for cell in cluster:
        for c in cycle_cells:
            if abs(c.y1 + c.y2 - cell.y1 - cell.y2) / (2 * height) <= 1 / (4 * nb_rows):
                matching_cells.append(cell)
                break

    # Try matching merged cells from cluster with cycle cell
    merged_cells = [(up, down) for up, down in zip(cluster, cluster[1:])
                    if len({up, down}.intersection(set(matching_cells))) == 0]
    for up, down in merged_cells:
        merged_cell = Cell(x1=up.x1, x2=up.x2, y1=min(up.y1, down.y1), y2=max(up.y2, down.y2))
        for c in cycle_cells:
            if abs(c.y1 + c.y2 - merged_cell.y1 - merged_cell.y2) / (2 * height) <= 1 / (4 * nb_rows):
                matching_cells += [up, down]
                break

    # Check if at least 75% of cells are matching with cluster
    return len(matching_cells) / len(cluster) >= 0.75


def merge_cycle_clusters(cycle: List[List[Cell]]) -> List[List[Cell]]:
    """
    Merge clusters in cycle that have common cells
    :param cycle: list of cell clusters
    :return: deduplicated cycle
    """
    # Merge clusters contained in cycle if they have common cells
    merged_cycle = list()
    for i in range(len(cycle)):
        for j in range(i, len(cycle)):
            # If clusters have common cells, find matching clusters
            if len(set(cycle[i]).intersection(cycle[j])) > 0:
                matching_clusters = [idx for idx, cl in enumerate(merged_cycle) if {i, j}.intersection(cl)]
                if matching_clusters:
                    remaining_clusters = [cl for idx, cl in enumerate(merged_cycle) if idx not in matching_clusters]
                    new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(merged_cycle) if idx in matching_clusters])
                    merged_cycle = remaining_clusters + [new_cluster]
                else:
                    merged_cycle.append({i, j})

    return [list(set([cell for idx in cl for cell in cycle[idx]]))
            for cl in merged_cycle]


def identify_tables(clusters: List[List[Cell]]) -> List[List[List[Cell]]]:
    """
    Identify corresponding clusters that can create a table
    :param clusters: list of cell clusters based on alignment
    :return: list of groups of corresponding clusters that can create a table
    """
    # Check if there are multiple clusters
    if len(clusters) <= 1:
        return []

    # Identify common rows between all cluster combinations
    d_common_rows = {(idx1, idx2): find_common_rows(cluster_1=clusters[idx1],
                                                    cluster_2=clusters[idx2])
                     for idx1, idx2 in list(combinations(range(len(clusters)), 2))
                     }

    # For each cluster, identify the list of best matching clusters
    dict_links = dict()
    for idx in range(len(clusters)):
        # Filter on links with the relevant clusters
        d_common_rows_cluster = {k: v for k, v in d_common_rows.items() if v > 1 and idx in k}
        # Get best links
        best_links = [k for k, v in d_common_rows_cluster.items() if v == max(d_common_rows_cluster.values())]

        for link in map(list, best_links):
            link.remove(idx)
            dict_links[idx] = dict_links.get(idx, []) + link

    # Identify cycles in graph, i.e matching clusters that can form a table
    cycles = get_maximal_cycles(cycles=[set(path) for node in dict_links
                                        for path in dfs(graph=dict_links, start=node, end=node)])

    # Check if there is at least one cycle
    if len(cycles) == 0:
        return []

    # Assert if other clusters are coherent with some cycles
    matching_cycles = list()
    for cycle in cycles:
        matching_cycle = cycle.copy()
        cycle = [clusters[idx] for idx in cycle]
        for idx, cl in enumerate(clusters):
            if match_with_cycle(cluster=clusters[idx], cycle=cycle):
                matching_cycle.add(idx)
        matching_cycles.append(matching_cycle)

    # Get final cycles by merging clusters that have cells in common in each cycle
    maximal_cycles = [[clusters[idx] for idx in cycle]
                      for cycle in get_maximal_cycles(cycles=matching_cycles)]
    final_cycles = [merge_cycle_clusters(cycle=cycle) for cycle in maximal_cycles]

    return final_cycles
