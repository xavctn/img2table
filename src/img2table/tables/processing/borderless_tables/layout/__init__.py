# coding: utf-8
from typing import List

import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.layout.column_segmentation import segment_image_columns
from img2table.tables.processing.borderless_tables.layout.segment_elements import get_segment_elements
from img2table.tables.processing.borderless_tables.layout.table_segments import get_table_segments
from img2table.tables.processing.borderless_tables.model import TableSegment


def segment_image(img: np.ndarray, lines: List[Line], char_length: float, median_line_sep: float,
                  contours: List[Cell]) -> List[TableSegment]:
    """
    Segment image and its elements
    :param img: image array
    :param lines: list of Line objects of the image
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param contours: list of image contours
    :return: list of ImageSegment objects with corresponding elements
    """
    # Segment image using columns
    column_segments = segment_image_columns(img=img,
                                            median_line_sep=median_line_sep,
                                            char_length=char_length,
                                            contours=contours)

    # Set segment elements
    column_segments = get_segment_elements(img=img,
                                           lines=lines,
                                           img_segments=column_segments,
                                           char_length=char_length,
                                           median_line_sep=median_line_sep,
                                           blur_size=3)

    # Within each column, identify segments that can correspond to tables
    tb_segments = [table_segment for col_segment in column_segments
                   for table_segment in get_table_segments(segment=col_segment,
                                                           char_length=char_length,
                                                           median_line_sep=median_line_sep)
                   ]

    return tb_segments
