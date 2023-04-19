# coding: utf-8
from typing import List, Optional

import cv2
import numpy as np
import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line


def threshold_dark_areas(img: np.ndarray, char_length: Optional[float]) -> np.ndarray:
    """
    Threshold image by differentiating areas with light and dark backgrounds
    :param img: image array
    :param char_length: average character length
    :return: threshold image
    """
    # Get threshold on image and binary image
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    binary_thresh = cv2.adaptiveThreshold(255 - blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # Mask on areas with dark background
    blur_size = int(2 * char_length) + 1 - int(2 * char_length) % 2 if char_length else 11
    blur = cv2.medianBlur(img, blur_size)
    mask = cv2.inRange(blur, 0, 100)

    # Get contours of dark areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each dark area, use binary threshold instead of regular threshold
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        margin = int(char_length)
        if min(w, h) > 2 * margin and w * h / np.prod(img.shape[:2]) < 0.9:
            thresh[y+margin:y+h-margin, x+margin:x+w-margin] = binary_thresh[y+margin:y+h-margin, x+margin:x+w-margin]

    return thresh


def overlapping_filter(lines: List[Line], max_gap: int = 5) -> List[Line]:
    """
    Process lines to merge close lines
    :param lines: lines
    :param max_gap: maximum gap used to merge lines
    :return: list of filtered lines
    """
    if len(lines) == 0:
        return []

    # Identify if lines are horizontal
    horizontal = all(map(lambda l: l.horizontal, lines))

    # If not horizontal, transpose all lines
    if not horizontal:
        lines = [line.transpose for line in lines]

    # Sort lines by secondary dimension
    lines = sorted(lines, key=lambda l: (l.y1, l.x1))

    # Create clusters of lines based on "similar" secondary dimension
    previous_sequence, current_sequence = iter(lines), iter(lines)
    line_clusters = [[next(current_sequence)]]
    for previous, line in zip(previous_sequence, current_sequence):
        # If the vertical difference between consecutive lines is too large, create a new cluster
        if line.y1 - previous.y1 > 2:
            # Large gap, we create a new empty sublist
            line_clusters.append([])

        # Append to last cluster
        line_clusters[-1].append(line)

    # Create final lines by "merging" lines within a cluster
    final_lines = list()
    for cluster in line_clusters:
        # Sort the cluster
        cluster = sorted(cluster, key=lambda l: min(l.x1, l.x2))

        # Loop over lines in the cluster to merge relevant lines together
        seq = iter(cluster)
        sub_clusters = [[next(seq)]]
        for line in seq:
            # If lines are vertically close, merge line with curr_line
            dim_2_sub_clust = max(map(lambda l: l.x2, sub_clusters[-1]))
            if line.x1 - dim_2_sub_clust <= max_gap:
                sub_clusters[-1].append(line)
            # If the difference in vertical coordinates is too large, create a new sub cluster
            else:
                sub_clusters.append([line])

        # Create lines from sub clusters
        for sub_cl in sub_clusters:
            y_value = round(np.average([l.y1 for l in sub_cl],
                                       weights=list(map(lambda l: l.length, sub_cl))))
            thickness = min(max(1, max(map(lambda l: l.y2, sub_cl)) - min(map(lambda l: l.y1, sub_cl))), 5)
            line = Line(x1=min(map(lambda l: l.x1, sub_cl)),
                        x2=max(map(lambda l: l.x2, sub_cl)),
                        y1=y_value,
                        y2=y_value,
                        thickness=thickness)

            if line.length > 0:
                final_lines.append(line)

    # If not horizontal, transpose all lines
    if not horizontal:
        final_lines = [line.transpose for line in final_lines]

    return final_lines


def remove_word_lines(lines: List[Line], contours: List[Cell]) -> List[Line]:
    """
    Remove lines that corresponds to contours in image
    :param lines: list of lines
    :param contours: list of image contours as cell objects
    :return: list of lines not intersecting with words
    """
    # Get contours dataframe
    df_cnts = pl.LazyFrame(data=[{"x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2} for c in contours])

    # If there are no lines or no contours, do nothing
    if len(lines) == 0 or df_cnts.collect(streaming=True).height == 0:
        return lines

    # Create dataframe containing lines
    df_lines = (pl.LazyFrame(data=[line.dict for line in lines])
                .with_columns([pl.max([pl.col('width'), pl.col('height')]).alias('length'),
                               (pl.col('x1') == pl.col('x2')).alias('vertical')]
                              )
                .with_row_count(name="line_id")
                .rename({"x1": "x1_line", "x2": "x2_line", "y1": "y1_line", "y2": "y2_line"})
                )

    # Merge both dataframes
    df_words_lines = df_cnts.join(df_lines, how='cross')

    # Compute intersection between contours bbox and lines
    # - vertical case
    vert_int = (
        (((pl.col('x1') + pl.col('x2')) / 2 - pl.col('x1_line')).abs() / (pl.col('x2') - pl.col('x1')) < 0.45)
        * pl.max([(pl.min([pl.col('y2'), pl.col('y2_line')]) - pl.max([pl.col('y1'), pl.col('y1_line')])), pl.lit(0)])
    )
    # - horizontal case
    hor_int = (
        (((pl.col('y1') + pl.col('y2')) / 2 - pl.col('y1_line')).abs() / (pl.col('y2') - pl.col('y1')) < 0.4)
        * pl.max([(pl.min([pl.col('x2'), pl.col('x2_line')]) - pl.max([pl.col('x1'), pl.col('x1_line')])), pl.lit(0)])
    )
    df_words_lines = df_words_lines.with_columns((pl.col('vertical') * vert_int
                                                  + (1 - pl.col('vertical')) * hor_int).alias('intersection')
                                                 )

    # Compute total intersection for each line
    df_inter = (df_words_lines.groupby(['line_id', 'length'])
                .agg(pl.col('intersection').sum().alias('intersection'))
                )

    # Identify lines that intersect contours
    intersecting_lines = (df_inter.filter(pl.col('intersection') / pl.col('length') > 0.5)
                          .collect(streaming=True)
                          .get_column('line_id')
                          .to_list()
                          )

    return [line for idx, line in enumerate(lines) if idx not in intersecting_lines]


def detect_lines(image: np.ndarray, contours: Optional[List[Cell]], char_length: Optional[float], rho: float = 1,
                 theta: float = np.pi / 180, threshold: int = 50, minLinLength: int = 290, maxLineGap: int = 6,
                 kernel_size: int = 20) -> (List[Line], List[Line]):
    """
    Detect horizontal and vertical lines on image
    :param image: image array
    :param contours: list of image contours as cell objects
    :param char_length: average character length
    :param rho: rho parameter for Hough line transform
    :param theta: theta parameter for Hough line transform
    :param threshold: threshold parameter for Hough line transform
    :param minLinLength: minLinLength parameter for Hough line transform
    :param maxLineGap: maxLineGap parameter for Hough line transform
    :param kernel_size: kernel size to filter on horizontal / vertical lines
    :return: horizontal and vertical lines
    """
    # Create copy of image
    img = image.copy()

    # Apply thresholding
    thresh = threshold_dark_areas(img=img, char_length=char_length)

    # Identify both vertical and horizontal lines
    for kernel_tup, gap in [((kernel_size, 1), 2 * maxLineGap), ((1, kernel_size), maxLineGap)]:
        # Apply masking on image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_tup)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Compute Hough lines on image and get lines
        hough_lines = cv2.HoughLinesP(mask, rho, theta, threshold, None, minLinLength, maxLineGap)

        # Handle case with no lines
        if hough_lines is None:
            yield []
            continue

        lines = [Line(*line[0].tolist()).reprocess() for line in hough_lines]

        # Remove lines that are not horizontal or vertical
        lines = [line for line in lines if line.horizontal or line.vertical]

        # Merge lines
        merged_lines = overlapping_filter(lines=lines, max_gap=gap)

        # If possible, remove lines that corresponds to words
        if contours is not None:
            merged_lines = remove_word_lines(lines=merged_lines, contours=contours)

        yield merged_lines
