# coding: utf-8
import operator
from typing import List

import cv2
import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.model import ImageSegment
from img2table.tables.processing.common import is_contained_cell, merge_contours


def create_image_segments(img: np.ndarray, median_line_sep: float) -> List[ImageSegment]:
    """
    Create segmentation of the image into specific parts
    :param img: image array
    :param median_line_sep: median line separation
    :return: list of image segments as Cell objects
    """
    # Reprocess images
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.Canny(blur, 0, 0)

    # Define kernel size by using median line separation
    kernel_size = round(median_line_sep / 3)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Create new image by adding black borders
    margin = 10
    back_borders = np.zeros(tuple(map(operator.add, img.shape, (2 * margin, 2 * margin))), dtype=np.uint8)
    back_borders[margin:img.shape[0] + margin, margin:img.shape[1] + margin] = dilate

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(back_borders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    return [ImageSegment(x1=seg.x1, y1=seg.y1, x2=seg.x2, y2=seg.y2) for seg in img_segments]


def get_segment_elements(img: np.ndarray, lines: List[Line], img_segments: List[ImageSegment], char_length: float,
                         median_line_sep: float, blur_size: int = 3) -> List[ImageSegment]:
    """
    Identify image elements that correspond to each segment
    :param img: image array
    :param lines: list of image lines
    :param img_segments: list of ImageSegment objects
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param blur_size: kernel size for blurring operation
    :return: list of ImageSegment objects with corresponding elements
    """
    # Reprocess image
    blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    thresh = cv2.Canny(blur, 85, 255)

    # Mask lines
    for l in lines:
        if l.horizontal:
            cv2.rectangle(thresh, (l.x1 - l.thickness, l.y1), (l.x2 + l.thickness, l.y2), (0, 0, 0), 3 * l.thickness)
        elif l.vertical:
            cv2.rectangle(thresh, (l.x1, l.y1 - l.thickness), (l.x2, l.y2 + l.thickness), (0, 0, 0), 2 * l.thickness)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(1.5 * char_length), int(median_line_sep // 6)))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Get list of contours
    elements = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        elements.append(Cell(x1=x, y1=y, x2=x + w, y2=y + h))

    # Merge elements
    elements = merge_contours(contours=elements,
                              vertically=None)

    # Get elements corresponding to each image segment
    for seg in img_segments:
        segment_elements = [cnt for cnt in elements if is_contained_cell(inner_cell=cnt, outer_cell=seg)]
        seg.set_elements(elements=segment_elements)

    return img_segments


def segment_image(img: np.ndarray, lines: List[Line], char_length: float, median_line_sep: float) -> List[ImageSegment]:
    """
    Segment image and its elements
    :param img: image array
    :param lines: list of Line objects of the image
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: list of ImageSegment objects with corresponding elements
    """
    # Create image segments
    image_segments = create_image_segments(img=img,
                                           median_line_sep=median_line_sep)

    # Detect elements corresponding to each segment
    image_segments = get_segment_elements(img=img,
                                          lines=lines,
                                          img_segments=image_segments,
                                          char_length=char_length,
                                          median_line_sep=median_line_sep,
                                          blur_size=3)

    return image_segments

