# coding: utf-8
from typing import List, NamedTuple

import numpy as np

from img2table.tables.objects.cell import Cell


class RowCenter(NamedTuple):
    y_center: int
    y_top: int
    y_bottom: int
    nb_cells: int
    cluster_id: int


# def get_row_delimiters(tb_clusters: List[List[Cell]], margin: int = 5) -> List[int]:
#     """
#     Identify row delimiters from clusters
#     :param tb_clusters: list of clusters composing the table
#     :param margin: margin used for extremities
#     :return: list of row delimiters
#     """
#     # Get all cells from clusters
#     cells = sorted([cell for cl in tb_clusters for cell in cl],
#                    key=lambda c: c.y1)
#
#     # Compute row clusters
#     seq = iter(cells)
#     row_clusters = [[next(seq)]]
#     for cell in seq:
#         cl_y1, cl_y2 = min([c.y1 for c in row_clusters[-1]]), max([c.y2 for c in row_clusters[-1]])
#         y_corr = min(cell.y2, cl_y2) - max(cell.y1, cl_y1)
#         if y_corr / max(cl_y2 - cl_y1, cell.y2 - cell.y1) <= 0.2:
#             row_clusters.append([])
#         row_clusters[-1].append(cell)
#
#     # Compute row bounds
#     row_bounds = [(min([c.y1 for c in row]), max([c.y2 for c in row])) for row in row_clusters]
#
#     # Create delimiters
#     y_delimiters = [round((up[1] + down[0]) / 2) for up, down in zip(row_bounds, row_bounds[1:])]
#     y_delimiters = [row_bounds[0][0] - margin] + y_delimiters + [row_bounds[-1][1] + margin]
#
#     return y_delimiters


def get_row_delimiters(tb_clusters: List[List[Cell]], margin: int = 5) -> List[int]:
    # Get row centers of each cluster and "merged" cells in cluster
    row_centers = list()
    for idx, cl in enumerate(tb_clusters):
        cl = sorted(cl, key=lambda c: c.y1 + c.y2)

        cl_center = [RowCenter(y_center=round((c.y1 + c.y2) / 2),
                               y_top=c.y1,
                               y_bottom=c.y2,
                               nb_cells=1,
                               cluster_id=idx)
                     for c in cl]
        cl_merged = [RowCenter(y_center=round((up.y1 + down.y2) / 2),
                               y_top=up.y1,
                               y_bottom=down.y2,
                               nb_cells=2,
                               cluster_id=idx)
                     for up, down in zip(cl, cl[1:])]

        row_centers += cl_center + cl_merged

    row_centers = sorted(row_centers, key=lambda x: (x.y_center, x.nb_cells))

    # Compute difference between consecutive elements of row_centers
    median_diff = np.mean([nextt.y_center - prev.y_center for prev, nextt in zip(row_centers, row_centers[1:])])
    print(median_diff)

    seq = iter(row_centers)
    r_clusters = [[next(seq)]]
    for r_center in seq:
        diff = r_center.y_center - r_clusters[-1][-1].y_center
        if diff > 0.75 * median_diff or any([c.cluster_id == r_center.cluster_id for c in r_clusters[-1]]):
            r_clusters.append([])
        r_clusters[-1].append(r_center)

    r_clusters = sorted(r_clusters, key=lambda cl: (len(cl),  -sum([c.nb_cells for c in cl])), reverse=True)

    for cl in r_clusters:
        print(cl)
    print('\n\n')

    seq = iter(r_clusters)
    f_clusters = [next(seq)]

    for r_cluster in seq:
        unique = True
        y_min = min([c.y_top for c in r_cluster])
        y_max = max([c.y_bottom for c in r_cluster])
        for cl in f_clusters:
            y_cl_min = min([c.y_top for c in cl])
            y_cl_max = max([c.y_bottom for c in cl])
            y_corr = min(y_max, y_cl_max) - max(y_min, y_cl_min)

            if y_corr / min(y_max - y_min, y_cl_max - y_cl_min) > 0.2:
                unique = False

        if unique:
            f_clusters.append(r_cluster)

    f_clusters = sorted(f_clusters, key=lambda cl: np.mean([c.y_center for c in cl]))
    for cl in f_clusters:
        print(cl)
    print('\n\n')

    f_clusters = [(min([c.y_top for c in cl]), max([c.y_bottom for c in cl])) for cl in f_clusters]

    row_dels = [round((prev[1] + nextt[0]) / 2) for prev, nextt in zip(f_clusters, f_clusters[1:])]
    row_dels = [f_clusters[0][0] - margin] + row_dels + [f_clusters[-1][1] + margin]

    return row_dels
