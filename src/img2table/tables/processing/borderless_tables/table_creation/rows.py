# coding: utf-8
from typing import List, NamedTuple, Tuple

import numpy as np

from img2table.tables.objects.cell import Cell


class RowCenter(NamedTuple):
    cells: List[Cell]
    cluster_id: int

    @property
    def y_top(self):
        return min([c.y1 for c in self.cells])

    @property
    def y_bottom(self):
        return max([c.y2 for c in self.cells])

    @property
    def y_center(self):
        return round((self.y_top + self.y_bottom) / 2)

    @property
    def nb_cells(self):
        return len(self.cells)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            try:
                assert self.cells == other.cells
                assert self.cluster_id == other.cluster_id
                return True
            except AssertionError:
                return False
        return False

    def __repr__(self):
        return f"RowCenter(y_center={self.y_center}, y_top={self.y_top}, y_bottom={self.y_bottom}, " \
               f"nb_cells={self.nb_cells}, cluster_id={self.cluster_id})"


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

def cluster_row_centers(row_centers: List[RowCenter]) -> List[List[RowCenter]]:
    # Sort row centers
    row_centers = sorted(row_centers, key=lambda x: (x.y_center, -x.nb_cells, x.cluster_id))

    # Compute difference between consecutive elements of row_centers
    max_diff = max(np.quantile(a=[nextt.y_center - prev.y_center for prev, nextt in zip(row_centers, row_centers[1:])],
                               q=0.4),
                   5)

    # Create clusters
    seq = iter(row_centers)
    r_clusters = [[next(seq)]]
    for r_center in seq:
        diff = r_center.y_center - r_clusters[-1][-1].y_center
        if diff > max_diff or any([c.cluster_id == r_center.cluster_id for c in r_clusters[-1]]):
            r_clusters.append([])
        r_clusters[-1].append(r_center)

    return r_clusters


def unify_clusters(cl_unique_r_center: List[List[RowCenter]],
                   cl_merged_r_center: List[List[RowCenter]]) -> List[List[RowCenter]]:
    # Identify if merged clusters wrongly override unique clusters
    relevant_merged_clusters = list()
    for merged_cluster in cl_merged_r_center:
        if merged_cluster in cl_unique_r_center:
            continue

        # Get cells corresponding to double clusters
        merged_cells = set([c for r_center in merged_cluster for c in r_center.cells])

        # Identify unique clusters that contain those cells
        matching_unique_clusters = [cl for cl in cl_unique_r_center
                                    if len(merged_cells.intersection([c for r_center in cl for c in r_center.cells])) > 0]

        if len(merged_cells) == len([c for cl in matching_unique_clusters for c in cl]):
            relevant_merged_clusters.append(merged_cluster)

    return cl_unique_r_center + relevant_merged_clusters


def row_delimiters_from_borders(borders: List[Tuple[int, int]], margin: int = 5) -> List[int]:
    # Get bottom and top delimiters
    top_del = borders[0][0] - margin
    bottom_del = borders[-1][1] + margin

    # Compute intermediate delimiters
    inter_dels = list()
    seq = iter(borders)
    cur_border = next(seq)
    for border in seq:
        y_corr = min(border[1], cur_border[1]) - max(border[0], cur_border[0])
        ref_height = min(border[1] - border[0], cur_border[1] - cur_border[0])
        if y_corr / ref_height < 0.5:
            inter_dels.append(round((cur_border[1] + border[0]) / 2))
            cur_border = border
        else:
            cur_border = (min(cur_border[0], border[0]), max(cur_border[1], border[1]))

    return [top_del] + inter_dels + [bottom_del]


def get_row_delimiters(tb_clusters: List[List[Cell]], margin: int = 5) -> List[int]:
    # Get row centers of each cluster and "merged" cells in cluster
    row_centers_unique = [RowCenter(cells=[c], cluster_id=idx) for idx, cl in enumerate(tb_clusters)
                          for c in cl]
    row_centers_double = [RowCenter(cells=[up, down], cluster_id=idx) for idx, cl in enumerate(tb_clusters)
                          for up, down in zip(sorted(cl, key=lambda c: c.y1 + c.y2),
                                              sorted(cl, key=lambda c: c.y1 + c.y2)[1:])
                          ]

    # Cluster unique row centers
    cl_unique_r_center = cluster_row_centers(row_centers=row_centers_unique)

    # Cluster both list of row centers
    cl_merged_r_center = cluster_row_centers(row_centers=row_centers_unique + row_centers_double)

    # Create unified clusters
    unified_clusters = unify_clusters(cl_unique_r_center=cl_unique_r_center,
                                      cl_merged_r_center=cl_merged_r_center)

    # Sort clusters by decreasing length
    sorted_clusters = sorted(unified_clusters,
                             key=lambda cl: (len(cl), -sum([c.nb_cells for c in cl])),
                             reverse=True)

    # Select final clusters by identify clusters containing unique cells
    seq = iter(sorted_clusters)
    f_clusters = [next(seq)]
    for r_cluster in seq:
        r_cluster_cells = [c for r_center in r_cluster for c in r_center.cells]
        f_clusters_cells = [c for cl in f_clusters for r_center in cl for c in r_center.cells]
        if len(set(f_clusters_cells).intersection(set(r_cluster_cells))) == 0:
            f_clusters.append(r_cluster)

    # Sort final clusters by vertical position
    f_clusters = sorted(f_clusters, key=lambda cl: np.mean([c.y_center for c in cl]))
    # Get borders of each final cluster
    f_clusters_borders = [(min([c.y_top for c in cl]), max([c.y_bottom for c in cl])) for cl in f_clusters]

    # Compute row delimiters
    row_dels = row_delimiters_from_borders(borders=f_clusters_borders, margin=margin)

    return row_dels
