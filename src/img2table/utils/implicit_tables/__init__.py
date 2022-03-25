# coding: utf-8
import copy
from typing import List

import numpy as np

from img2table.objects.tables import Cell, Row, Table
from img2table.utils.common import get_bounding_area_text
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
    # Create table that represents the entire image
    height, width, _ = white_img.shape
    cell = Cell(0, 0, width, height)
    img_table = Table(rows=Row(cells=cell))

    # Get bounding boxes / contours in table
    img_table = get_bounding_area_text(img=copy.deepcopy(white_img),
                                       table=img_table,
                                       margin=0,
                                       blur_size=5,
                                       kernel_size=5,
                                       merge_vertically=None)

    # Cluster contours
    contours = [cell for row in img_table.items for cell in row.contours]
    clusters = cluster_contours(contours=contours, tables=tables)

    # Create group of clusters that correspond to tables
    cluster_groups = group_clusters(clusters=clusters)

    # Convert cluster groups to tables
    tables = [cluster_group_to_table(cluster_group=cluster_group) for cluster_group in cluster_groups]

    return tables
