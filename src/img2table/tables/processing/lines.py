# coding: utf-8
from typing import List

import cv2
import numpy as np

from img2table.tables.objects.line import Line


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

    # Define axis of analysis
    main_dim_1 = "x1" if horizontal else "y1"
    main_dim_2 = "x2" if horizontal else "y2"
    sec_dim_1 = "y1" if horizontal else "x1"
    sec_dim_2 = "y2" if horizontal else "x2"

    # Sort lines by secondary dimension
    lines = sorted(lines, key=lambda l: (getattr(l, sec_dim_1), getattr(l, main_dim_1)))

    # Create clusters of lines based on "similar" secondary dimension
    previous_sequence, current_sequence = iter(lines), iter(lines)
    line_clusters = [[next(current_sequence)]]
    for previous, line in zip(previous_sequence, current_sequence):
        # If the vertical difference between consecutive lines is too large, create a new cluster
        if getattr(line, sec_dim_1) - getattr(previous, sec_dim_1) > 2:
            # Large gap, we create a new empty sublist
            line_clusters.append([])

        # Append to last cluster
        line_clusters[-1].append(line)

    # Create final lines by "merging" lines within a cluster
    final_lines = list()
    for cluster in line_clusters:
        # Sort the cluster
        cluster = sorted(cluster, key=lambda l: min(getattr(l, main_dim_1), getattr(l, main_dim_2)))

        # Loop over lines in the cluster to merge relevant lines together
        seq = iter(cluster)
        sub_clusters = [[next(seq)]]
        for line in seq:
            # If lines are vertically close, merge line with curr_line
            dim_2_sub_clust = max(map(lambda l: getattr(l, main_dim_2), sub_clusters[-1]))
            if getattr(line, main_dim_1) - dim_2_sub_clust <= max_gap:
                sub_clusters[-1].append(line)
            # If the difference in vertical coordinates is too large, create a new sub cluster
            else:
                sub_clusters.append([line])

        # Create lines from sub clusters
        for sub_cl in sub_clusters:
            sec_dim = round(np.average([getattr(l, sec_dim_1) for l in sub_cl],
                                       weights=list(map(lambda l: l.length, sub_cl))))
            line_dict = {
                main_dim_1: min(map(lambda l: getattr(l, main_dim_1), sub_cl)),
                main_dim_2: max(map(lambda l: getattr(l, main_dim_2), sub_cl)),
                sec_dim_1: sec_dim,
                sec_dim_2: sec_dim
            }
            line = Line(**line_dict)

            if line.length > 0:
                final_lines.append(line)

    return final_lines


def detect_lines(image: np.ndarray, rho: float = 1, theta: float = np.pi / 180, threshold: int = 50,
                 minLinLength: int = 290, maxLineGap: int = 6, kernel_size: int = 20) -> (List[Line], List[Line]):
    """
    Detect horizontal and vertical lines on image
    :param image: image array
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

    # Apply blurring and thresholding
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # Identify both vertical and horizontal lines
    for kernel_tup, gap in [((kernel_size, 1), 2 * maxLineGap), ((1, kernel_size), maxLineGap)]:
        # Apply masking on image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_tup)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Compute Hough lines on image and get lines
        hough_lines = cv2.HoughLinesP(mask, rho, theta, threshold, None, minLinLength, maxLineGap)
        lines = [Line(*line[0].tolist()).reprocess() for line in hough_lines]

        # Merge lines
        merged_lines = overlapping_filter(lines=lines, max_gap=gap)
        yield merged_lines
