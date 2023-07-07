# coding: utf-8
from typing import List

import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.image_segmentation.column_segmentation import segment_image_columns
from img2table.tables.processing.borderless_tables.image_segmentation.segment_elements import get_segment_elements
from img2table.tables.processing.borderless_tables.image_segmentation.segmentation import create_image_segments
from img2table.tables.processing.borderless_tables.model import ImageSegment


def segment_image(img: np.ndarray, lines: List[Line], char_length: float, median_line_sep: float,
                  contours: List[Cell]) -> List[ImageSegment]:
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

    # Identify segments in image
    img_segments = [seg for col_seg in column_segments
                    for seg in create_image_segments(img=img,
                                                     area=col_seg,
                                                     median_line_sep=median_line_sep,
                                                     char_length=char_length)]

    # Detect elements corresponding to each segment
    img_segments = get_segment_elements(img=img,
                                        lines=lines,
                                        img_segments=img_segments,
                                        char_length=char_length,
                                        median_line_sep=median_line_sep,
                                        blur_size=3)

    return img_segments
