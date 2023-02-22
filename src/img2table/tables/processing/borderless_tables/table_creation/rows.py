# coding: utf-8
from typing import List, NamedTuple, Tuple

import numpy as np

from img2table.tables.objects.cell import Cell


class CellsCenter(NamedTuple):
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
        return f"CellsCenter(y_center={self.y_center}, y_top={self.y_top}, y_bottom={self.y_bottom}, " \
               f"nb_cells={self.nb_cells}, cluster_id={self.cluster_id})"


def merge_aligned_in_column(tb_clusters: List[List[Cell]]) -> List[List[Cell]]:
    """
    Merge cells that corresponds to the same line within each column clusters
    :param tb_clusters: list of column clusters corresponding to a table
    :return: list of processed column clusters corresponding to a table
    """
    merged_col_clusters = list()
    
    # Loop over each column
    for col_cluster in tb_clusters:
        # Sort cluster
        col_cluster = sorted(col_cluster, key=lambda c: c.y1 + c.y2)
        
        # Identify cells in cluster that corresponds to the same row and merge them
        seq = iter(col_cluster)
        row_groups = [[next(seq)]]
        for cell in seq:
            # Compute difference of cell centers
            y_diff = abs(cell.y1 + cell.y2 - row_groups[-1][-1].y1 - row_groups[-1][-1].y2) / 2
            if y_diff > 5:
                row_groups.append([])
            row_groups[-1].append(cell)
        
        # Recreate new column cluster
        new_col = [Cell(x1=min([c.x1 for c in gp]),
                        y1=min([c.y1 for c in gp]),
                        x2=max([c.x2 for c in gp]),
                        y2=max([c.y2 for c in gp])) 
                   for gp in row_groups]
        
        merged_col_clusters.append(new_col)
        
    return merged_col_clusters
    
    
def cluster_cells_centers(cells_centers: List[CellsCenter]) -> List[List[CellsCenter]]:
    """
    Create clusters of CellsCenter objects based on vertical alignment and correspondence
    :param cells_centers: list of CellsCenter objects
    :return: clusters of CellsCenter objects based on vertical alignment and correspondence
    """
    # Sort cells centers
    cells_centers = sorted(cells_centers, key=lambda x: (x.y_center, -x.nb_cells, x.cluster_id))

    # Compute difference between consecutive elements of cells_centers
    max_diff = max(np.quantile(a=[nextt.y_center - prev.y_center
                                  for prev, nextt in zip(cells_centers, cells_centers[1:])],
                               q=0.4),
                   5)

    # Create clusters
    seq = iter(cells_centers)
    c_clusters = [[next(seq)]]
    for c_center in seq:
        # Compute vertical difference with previous CellsCenter
        diff = c_center.y_center - c_clusters[-1][-1].y_center
        if diff > max_diff or any([c.cluster_id == c_center.cluster_id for c in c_clusters[-1]]):
            c_clusters.append([])
        c_clusters[-1].append(c_center)

    return c_clusters


def unify_clusters(cl_unique_c_center: List[List[CellsCenter]],
                   cl_merged_c_center: List[List[CellsCenter]]) -> List[List[CellsCenter]]:
    """
    Unify clusters created from unique cells and clusters created from merged cells
    :param cl_unique_c_center: CellsCenter clusters created from unique cells
    :param cl_merged_c_center: CellsCenter clusters created from merged cells
    :return: coherent CellsCenter clusters from both unique and merged cells
    """
    # Identify if merged clusters are coherent, i.e can be created from a set of unique clusters
    relevant_merged_clusters = list()
    for merged_cluster in cl_merged_c_center:
        # If the merged cluster is also in unique clusters, continue
        if merged_cluster in cl_unique_c_center:
            continue

        # Get cells corresponding to merged cluster
        merged_cells = set([c for r_center in merged_cluster for c in r_center.cells])

        # Identify unique clusters that contain those cells
        matching_unique_clusters = [cl for cl in cl_unique_c_center
                                    if len(merged_cells.intersection([c for r_center in cl for c in r_center.cells])) > 0]

        # Assess if cells in merged clusters are the same as in all matching unique clusters
        if len(merged_cells) >= len([c for cl in matching_unique_clusters for c in cl]):
            relevant_merged_clusters.append(merged_cluster)

    return cl_unique_c_center + relevant_merged_clusters


def row_delimiters_from_borders(borders: List[Tuple[int, int]], margin: int = 5) -> List[int]:
    """
    Create row delimiters from clusters borders
    :param borders: list of cluster top and bottom borders
    :param margin: margin used for top and bottom cells
    :return: list of row delimiters
    """
    # Get bottom and top delimiters
    top_del = borders[0][0] - margin
    bottom_del = borders[-1][1] + margin

    # Compute intermediate delimiters
    inter_dels = list()
    seq = iter(borders)
    cur_border = next(seq)
    for border in seq:
        # Check vertical correlation with previous border
        y_corr = min(border[1], cur_border[1]) - max(border[0], cur_border[0])
        ref_height = min(border[1] - border[0], cur_border[1] - cur_border[0])
        if y_corr / ref_height < 0.5:
            # If vertical positions do not correspond, create a new row delimiter
            inter_dels.append(round((cur_border[1] + border[0]) / 2))
            cur_border = border
        else:
            # If vertical positions correspond, update border by merging both borders
            cur_border = (min(cur_border[0], border[0]), max(cur_border[1], border[1]))

    return [top_del] + inter_dels + [bottom_del]


def get_row_delimiters(tb_clusters: List[List[Cell]], margin: int = 5) -> List[int]:
    """
    Identify row delimiters from clusters
    :param tb_clusters: list of clusters composing the table
    :param margin: margin used for extremities
    :return: list of row delimiters
    """
    # Merge cells that corresponds to the same line within each column clusters
    tb_clusters = merge_aligned_in_column(tb_clusters=tb_clusters)

    # Get cells centers of each cluster and "merged" cells in cluster
    cells_centers_unique = [CellsCenter(cells=[c], cluster_id=idx) for idx, cl in enumerate(tb_clusters)
                            for c in cl]
    cells_centers_double = [CellsCenter(cells=[up, down], cluster_id=idx) for idx, cl in enumerate(tb_clusters)
                            for up, down in zip(sorted(cl, key=lambda c: c.y1 + c.y2),
                                                sorted(cl, key=lambda c: c.y1 + c.y2)[1:])
                            ]

    # Cluster unique cells centers
    cl_unique_c_center = cluster_cells_centers(cells_centers=cells_centers_unique)

    # Cluster both list of cells centers
    cl_merged_c_center = cluster_cells_centers(cells_centers=cells_centers_unique + cells_centers_double)

    # Create unified clusters
    unified_clusters = unify_clusters(cl_unique_c_center=cl_unique_c_center,
                                      cl_merged_c_center=cl_merged_c_center)

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
