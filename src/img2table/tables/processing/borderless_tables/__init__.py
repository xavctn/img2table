# coding: utf-8
from typing import List

import numpy as np

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.alignment import cluster_aligned_text
from img2table.tables.processing.borderless_tables.identify_tables import identify_tables
from img2table.tables.processing.borderless_tables.segment_image import segment_image_text
from img2table.tables.processing.borderless_tables.table_creation import create_table_from_clusters
from img2table.tables.processing.common import is_contained_cell


def deduplicate_tables(identified_tables: List[Table], existing_tables: List[Table]) -> List[Table]:
    """
    Deduplicate identified borderless tables with already identified tables in order to avoid duplicates and overlap
    :param identified_tables: list of borderless tables identified
    :param existing_tables: list of already identified tables
    :return: deduplicated list of identified borderless tables
    """
    # Sort tables by area
    identified_tables = sorted(identified_tables, key=lambda tb: tb.area, reverse=True)

    # For each table check if it does not overlap with an existing table
    final_tables = list()
    for table in identified_tables:
        if not any([is_contained_cell(inner_cell=table.cell, outer_cell=tb.cell, percentage=0.5)
                    for tb in existing_tables + final_tables]):
            final_tables.append(table)

    return final_tables


def detect_borderless_tables(img: np.ndarray, ocr_df: OCRDataframe, existing_tables: List[Table]) -> List[Table]:
    """
    Identify borderless tables in image
    :param img: image array
    :param ocr_df: OCRDataframe
    :param existing_tables: list of already identified table objects
    :return: list of borderless tables identified in image
    """
    # Segment image and get text contours corresponding to each segment
    image_segments = segment_image_text(img=img, ocr_df=ocr_df)

    # Identify tables in each segment
    list_tables = list()
    for segment in image_segments:
        # Cluster text contours based on alignment
        alignment_clusters = cluster_aligned_text(segment=segment)

        # Table detection based on clusters
        table_clusters = identify_tables(clusters=alignment_clusters)

        # Create table objects
        for table_cluster in table_clusters:
            table = create_table_from_clusters(tb_clusters=table_cluster,
                                               segment_cells=[c for c in segment])

            if table:
                list_tables.append(table)

    return deduplicate_tables(identified_tables=list_tables,
                              existing_tables=existing_tables)

