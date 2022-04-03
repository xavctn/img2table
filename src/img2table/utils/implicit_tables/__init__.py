# coding: utf-8
from typing import List

import numpy as np

from img2table.objects.tables import Cell, Table
from img2table.utils.common import get_contours_cell
from img2table.utils.implicit_tables.cluster_groups import group_clusters
from img2table.utils.implicit_tables.clusters import cluster_contours
from img2table.utils.implicit_tables.group_to_table import cluster_group_to_table


def detect_implicit_tables(white_img: np.ndarray, tables: List[Table]) -> List[Table]:
    """
    Detect and create Table object for implicit tables
    :param white_img: image array
    :param tables: list of existing Table object in image
    :return: list of Table objects corresponding to implicit tables
    """
    # Create cell that represents the entire image
    height, width, _ = white_img.shape
    cell = Cell(0, 0, width, height)

    # Get bounding boxes / contours in cell
    contours = get_contours_cell(img=white_img,
                                 cell=cell,
                                 margin=0,
                                 blur_size=5,
                                 kernel_size=5,
                                 merge_vertically=None)

    # Cluster contours
    clusters = cluster_contours(contours=contours, tables=tables)

    # Create group of clusters that correspond to tables
    cluster_groups = group_clusters(clusters=clusters)

    # Convert cluster groups to tables
    implicit_tables = [cluster_group_to_table(cluster_group=cluster_group) for cluster_group in cluster_groups]

    return implicit_tables
