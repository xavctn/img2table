# coding: utf-8
import operator
from typing import List, Dict

import cv2
import numpy as np
import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.common import is_contained_cell, merge_contours


def create_image_segments(img: np.ndarray, ocr_df: OCRDataframe) -> List[Cell]:
    """
    Create segmentation of the image into specific parts
    :param img: image array
    :param ocr_df: OCRDataframe object
    :return: list of image segments as Cell objects
    """
    # Reprocess images
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.Canny(blur, 0, 0)

    # Define kernel size by using most stable metric between different OCRs
    if ocr_df.median_line_sep is not None:
        kernel_size = round(ocr_df.median_line_sep / 3)
    elif ocr_df.text_size is not None:
        kernel_size = round(ocr_df.text_size / 2)
    else:
        kernel_size = 20

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

    return img_segments


def text_countours_from_group(word_group: List[Dict]) -> List[Cell]:
    """
    Identify text contours from list of words
    :param word_group: list of words represented by bounding boxes
    :return: list of Cell objects representing text contours
    """
    # Sort words
    word_group = sorted(word_group, key=lambda c: (c.get('y1') + c.get('y2'), c.get('x1')))

    # Separate words into lines
    previous_seq, current_seq = iter(word_group), iter(word_group)
    lines = [[next(current_seq)]]
    for current, previous in zip(current_seq, previous_seq):
        y_corr = min(current.get('y2'), previous.get('y2')) - max(current.get('y1'), previous.get('y1'))
        height = min(current.get('y2') - current.get('y1'), previous.get('y2') - previous.get('y1'))
        if y_corr / height <= 0.25:
            lines.append([])
        lines[-1].append(current)

    # Create text contours from each line
    text_contours = list()
    for line in lines:
        if len(line) == 1:
            cell = line.pop()
            cell.pop('length')
            text_contours.append(Cell(**cell))
            continue

        line = sorted(line, key=lambda c: c.get('x1'))
        median_word_sep = np.median([max(w2.get('x1') - w1.get('x2'), 0) for w1, w2 in zip(line, line[1:])])

        seq = iter(line)
        word_cnts = [[next(seq)]]
        for word in seq:
            prev_word = word_cnts[-1][-1]
            x_diff = word.get('x1') - prev_word.get('x2')
            max_diff = min(3 * median_word_sep,
                           2 * min((word.get('x2') - word.get('x1')) // word.get('length'),
                                   (prev_word.get('x2') - prev_word.get('x1')) // prev_word.get('length')
                                   )
                           )
            if x_diff > max_diff:
                word_cnts.append([])
            word_cnts[-1].append(word)

        for word_cnt in word_cnts:
            cnt = Cell(x1=min([w.get('x1') for w in word_cnt]),
                       y1=min([w.get('y1') for w in word_cnt]),
                       x2=max([w.get('x2') for w in word_cnt]),
                       y2=max([w.get('y2') for w in word_cnt]))
            text_contours.append(cnt)

    return text_contours


def create_word_contours(ocr_df: OCRDataframe) -> List[Cell]:
    """
    Create list of contours corresponding to text present in the image
    :param ocr_df: OCRDataframe object
    :return: list of text contours as Cell objects
    """
    # Get list of words groups by parent
    list_word_groups = (ocr_df.df
                        # Filter on relevant words from OCR
                        .filter(pl.col('class') == "ocrx_word")
                        .filter(pl.col('value').is_not_null())
                        .filter(pl.col('confidence') >= 20)
                        .with_columns(pl.col('value').str.n_chars().alias('length'))
                        # Aggregate by parent
                        .groupby('parent')
                        .agg(pl.struct(['x1', 'y1', 'x2', 'y2', 'length']).alias('words'))
                        .collect()
                        .to_dicts()
                        )

    # Create contours corresponding to text in image
    text_contours = [cnt for word_group in list_word_groups
                     for cnt in text_countours_from_group(word_group=word_group.get('words'))]

    return text_contours


def segment_image_text(img: np.ndarray, ocr_df: OCRDataframe) -> List[List[Cell]]:
    """
    Create word contours and group them within image segments
    :param img: image array
    :param ocr_df: OCRDataframe object
    :return: list of image segments with associated text contours
    """
    # Create segmentation of the image into specific parts
    img_segments = create_image_segments(img=img, ocr_df=ocr_df)

    # Create list of contours corresponding to text present in the image
    text_contours = create_word_contours(ocr_df=ocr_df)

    dict_segments = {seg: [] for seg in img_segments}
    for cnt in text_contours:
        # Find most likely segment
        matching_segments = sorted([seg for seg in img_segments if is_contained_cell(inner_cell=cnt, outer_cell=seg)],
                                   key=lambda s: s.area,
                                   reverse=True)
        if matching_segments:
            best_segment = matching_segments.pop(0)
            dict_segments[best_segment].append(cnt)

    return list(dict_segments.values())
