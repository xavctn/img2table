# coding: utf-8
from typing import List

import numpy as np

from img2table.document import Image
from img2table.ocr import TesseractOCR
from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.alignment import cluster_aligned_text
from img2table.tables.processing.borderless_tables.identify_tables import identify_tables
from img2table.tables.processing.borderless_tables.segment_image import segment_image_text
from img2table.tables.processing.borderless_tables.table_creation import create_table_from_clusters


def identify_borderless_tables(img: np.ndarray, ocr_df: OCRDataframe) -> List[Table]:
    """
    Identify borderless tables in image
    :param img: image array
    :param ocr_df: OCRDataframe
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

    return list_tables


if __name__ == '__main__':
    img = Image(r"C:\Users\xavca\Pictures\test_6.png")
    ocr = TesseractOCR()
    ocr_df = ocr.of(img)
    img = list(img.images)[0]

    tables = identify_borderless_tables(img=img, ocr_df=ocr_df)

    print(tables)