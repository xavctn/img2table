# coding: utf-8
from typing import List

import numpy as np

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.column_delimiters import get_whitespace_column_delimiters
from img2table.tables.processing.borderless_tables.lines import identify_line_groups
from img2table.tables.processing.borderless_tables.segment_image import segment_image
from img2table.tables.processing.borderless_tables.table import identify_table
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
        if not any([max(is_contained_cell(inner_cell=table.cell, outer_cell=tb.cell, percentage=0.1),
                        is_contained_cell(inner_cell=tb.cell, outer_cell=table.cell, percentage=0.1))
                    for tb in existing_tables + final_tables]):
            final_tables.append(table)

    return final_tables


def identify_borderless_tables(img: np.ndarray, lines: List[Line], ocr_df: OCRDataframe,
                               existing_tables: List[Table]) -> List[Table]:
    # Segment image
    img_segments = segment_image(img=img,
                                 lines=lines,
                                 ocr_df=ocr_df)

    # In each segment, create groups of lines and identify tables
    tables = list()
    for seg in img_segments:
        # Identify line groups in segment
        seg_line_groups = identify_line_groups(segment=seg,
                                               ocr_df=ocr_df)

        # For each line group, identify column delimiters and create tables
        for line_gp in seg_line_groups.line_groups:
            # Get column delimiters
            col_delimiters = get_whitespace_column_delimiters(line_group=line_gp,
                                                              segment_elements=seg_line_groups.elements)

            # Create table
            table = identify_table(line_group=line_gp,
                                   column_delimiters=col_delimiters,
                                   lines=lines,
                                   elements=seg_line_groups.elements)

            if table:
                tables.append(table)

    return deduplicate_tables(identified_tables=tables,
                              existing_tables=existing_tables)
