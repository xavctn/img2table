# coding: utf-8
from typing import List

import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.common import get_contours_cell, is_contained_cell


def create_image_segments(img: np.ndarray, dpi: int = 200) -> List[Cell]:
    """
    Create segmentation of the image into specific parts
    :param img: image array
    :param dpi: estimated dpi of the image
    :return: list of image segments as Cell objects
    """
    # Segmentation of image into "large" parts
    img_segments = get_contours_cell(img=img,
                                     cell=Cell(x1=0, y1=0, x2=img.shape[1], y2=img.shape[0]),
                                     margin=0,
                                     blur_size=3,
                                     kernel_size=dpi // 10,
                                     merge_vertically=True)

    return img_segments


def create_word_contours(img: np.ndarray, dpi: int = 200) -> List[Cell]:
    """
    Create list of contours corresponding to text present in the image
    :param img: image array
    :param dpi: estimated dpi of the image
    :return: list of text contours as Cell objects
    """
    # Create contours corresponding to text in image
    text_contours = get_contours_cell(img=img,
                                      cell=Cell(x1=0, y1=0, x2=img.shape[1], y2=img.shape[0]),
                                      margin=0,
                                      blur_size=3,
                                      kernel_size=dpi * 3 // 200,
                                      merge_vertically=None)

    return text_contours


def segment_image_text(img: np.ndarray, dpi: int = 200) -> List[List[Cell]]:
    """
    Create word contours and group them within image segments
    :param img: image array
    :param dpi: estimated dpi of the image
    :return: list of image segments with associated text contours
    """
    # Create segmentation of the image into specific parts
    img_segments = create_image_segments(img=img, dpi=dpi)

    # Create list of contours corresponding to text present in the image
    text_contours = create_word_contours(img=img, dpi=dpi)

    dict_segments = {seg: [] for seg in img_segments}
    for cnt in text_contours:
        # Find most likely segment
        best_segment = sorted([seg for seg in img_segments if is_contained_cell(inner_cell=cnt, outer_cell=seg)],
                              key=lambda s: s.width * s.height,
                              reverse=True).pop(0)
        dict_segments[best_segment].append(cnt)

    return list(dict_segments.values())
