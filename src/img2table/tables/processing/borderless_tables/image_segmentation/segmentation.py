# coding: utf-8
import operator
from typing import List

import cv2
import numpy as np

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import ImageSegment
from img2table.tables.processing.common import merge_contours


def create_image_segments(img: np.ndarray, area: Cell, median_line_sep: float, char_length: float) -> List[ImageSegment]:
    """
    Create segmentation of the image into specific parts
    :param img: image array
    :param area: Cell indicating area of interest
    :param median_line_sep: median line separation
    :param char_length: average character length
    :return: list of image segments as Cell objects
    """
    # Crop image
    cropped_img = img[area.y1:area.y2, area.x1:area.x2]

    # Reprocess images
    blur = cv2.medianBlur(cropped_img, 5)
    thresh = cv2.Canny(blur, 0, 0)

    # Define kernel by using median line separation and character length
    kernel_size = (max(int(2 * char_length), int(round(median_line_sep / 3)), 1),
                   max(int(round(median_line_sep / 3)), 1))

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Create new image by adding black borders
    margin = 10
    black_borders = np.zeros(tuple(map(operator.add, (area.height, area.width), (2 * margin, 2 * margin))), dtype=np.uint8)
    black_borders[margin:area.height + margin, margin:area.width + margin] = dilate

    # Find contours
    cnts = cv2.findContours(black_borders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Get list of contours
    list_cnts_cell = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x = x - margin
        y = y - margin
        contour_cell = Cell(x, y, x + w, y + h)
        list_cnts_cell.append(contour_cell)

    # Add contours to row
    img_segments = merge_contours(contours=list_cnts_cell,
                                  vertically=None)

    # Merge segments that are vertically coherent
    cl_f = lambda s1, s2: min(s1.y2, s2.y2) - max(s1.y1, s2.y1) >= 0.5 * min(s1.height, s2.height)
    segment_groups = cluster_items(items=img_segments,
                                   clustering_func=cl_f)

    return [ImageSegment(x1=min([seg.x1 for seg in seg_gp]) + area.x1,
                         y1=min([seg.y1 for seg in seg_gp]) + area.y1,
                         x2=max([seg.x2 for seg in seg_gp]) + area.x1,
                         y2=max([seg.y2 for seg in seg_gp]) + area.y1)
            for seg_gp in segment_groups]
