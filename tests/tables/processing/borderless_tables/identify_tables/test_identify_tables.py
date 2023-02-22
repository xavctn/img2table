# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.identify_tables import find_common_rows, dfs, get_maximal_cycles, \
    merge_cycle_clusters, identify_tables, match_with_cycle


def test_find_common_rows():
    cluster_1 = [Cell(x1=0, x2=10, y1=0, y2=10),
                 Cell(x1=0, x2=10, y1=50, y2=60),
                 Cell(x1=0, x2=10, y1=100, y2=110)]

    cluster_2 = [Cell(x1=100, x2=110, y1=0, y2=10),
                 Cell(x1=100, x2=110, y1=200, y2=250),
                 Cell(x1=100, x2=110, y1=100, y2=110)]

    result = find_common_rows(cluster_1=cluster_1, cluster_2=cluster_2)

    assert result == 2


def test_dfs():
    graph = {1: [3, 4],
             2: [1, 5],
             3: [4],
             4: [2],
             5: [4, 2]}

    assert list(dfs(graph=graph, start=1, end=1)) == [[4, 2, 1], [3, 4, 2, 1]]
    assert list(dfs(graph=graph, start=1, end=5)) == [[4, 2, 5], [3, 4, 2, 5]]


def test_get_maximal_cycles():
    cycles = [{1, 2, 3}, {1, 2, 3, 4}, {5, 6}, {1, 3, 4, 6}]

    result = get_maximal_cycles(cycles=cycles)

    assert result == [{1, 2, 3, 4}, {1, 3, 4, 6}, {5, 6}]


def test_match_with_cycle():
    cluster = [Cell(x1=0, x2=0, y1=0, y2=10), Cell(x1=0, x2=0, y1=20, y2=30),
               Cell(x1=0, x2=0, y1=30, y2=40), Cell(x1=0, x2=0, y1=100, y2=110)]
    cycle = [[Cell(x1=0, x2=0, y1=0, y2=10)],
             [Cell(x1=0, x2=0, y1=20, y2=40)],
             [Cell(x1=0, x2=0, y1=100, y2=110)]]

    assert match_with_cycle(cluster=cluster, cycle=cycle)

    # Pop from cycle
    cycle.pop(1)
    assert not match_with_cycle(cluster=cluster, cycle=cycle)


def test_merge_cycle_clusters():
    cycle = [
        [Cell(x1=0, x2=10, y1=0, y2=10), Cell(x1=0, x2=10, y1=10, y2=20)],
        [Cell(x1=100, x2=110, y1=0, y2=10), Cell(x1=0, x2=10, y1=10, y2=20)],
        [Cell(x1=98, x2=137, y1=0, y2=10), Cell(x1=56, x2=87, y1=10, y2=20)],
    ]

    result = merge_cycle_clusters(cycle=cycle)

    expected = [
        [Cell(x1=100, x2=110, y1=0, y2=10), Cell(x1=0, x2=10, y1=0, y2=10), Cell(x1=0, x2=10, y1=10, y2=20)],
        [Cell(x1=98, x2=137, y1=0, y2=10), Cell(x1=56, x2=87, y1=10, y2=20)],
    ]

    assert [set(cycle) for cycle in result] == [set(cycle) for cycle in expected]


def test_identify_tables():
    with open("test_data/clusters.json", "r") as f:
        clusters = [[Cell(**element) for element in cl] for cl in json.load(f)]

    result = identify_tables(clusters=clusters)

    with open("test_data/expected.json", "r") as f:
        expected = [[[Cell(**element) for element in cl] for cl in tb] for tb in json.load(f)]

    assert [[set(cl) for cl in tb] for tb in result] == [[set(cl) for cl in tb] for tb in expected]
