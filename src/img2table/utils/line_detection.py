# coding: utf-8
import statistics
from typing import List

import numpy as np
from cv2 import cv2

from img2table.objects.tables import Line


def overlapping_filter(lines: List[Line], horizontal: bool = True, max_gap: int = 5) -> List[Line]:
    """
    Process lines to merge close lines
    :param lines: lines
    :param horizontal: boolean indicating if horizontal lines are processed
    :param max_gap: maximum gap used to merge lines
    :return: list of filtered lines
    """
    if horizontal:
        main_dim_1 = "x1"
        main_dim_2 = "x2"
        sec_dim_1 = "y1"
        sec_dim_2 = "y2"
    else:
        main_dim_1 = "y1"
        main_dim_2 = "y2"
        sec_dim_1 = "x1"
        sec_dim_2 = "x2"

    if len(lines) == 0:
        return []

    # Sort lines by secondary dimension
    lines = sorted(lines, key=lambda l: (getattr(l, sec_dim_1), getattr(l, main_dim_1)))

    # Create clusters of lines based on "similar" secondary dimension
    line_clusters = list()
    for idx, line in enumerate(lines):
        if idx == 0:
            curr_cluster = [line]
        else:
            # Compute vertical difference between consecutive lines
            diff_index = getattr(line, sec_dim_1) - getattr(curr_cluster[0], sec_dim_1)
            # If the difference is too large, add curr_cluster to list clusters and set new cluster with the current
            # line
            if diff_index > max_gap:
                line_clusters.append(curr_cluster)
                curr_cluster = [line]
            # Otherwise, set line coordinates to coherent cluster values and append line to current cluster
            else:
                curr_cluster += [line]
    line_clusters.append(curr_cluster)

    # Create final lines by "merging" lines within a cluster
    final_lines = list()
    for cluster in line_clusters:
        # Sort the cluster
        cluster = sorted(cluster, key=lambda l: min(getattr(l, main_dim_1), getattr(l, main_dim_2)))

        # Loop over lines in the cluster to merge relevant lines together
        for idx, line in enumerate(cluster):
            if idx == 0:
                sub_cluster = [line]
            else:
                # If lines are vertically close, merge line with curr_line
                if getattr(line, main_dim_1) <= getattr(sub_cluster[-1], main_dim_2) + max_gap:
                    sub_cluster.append(line)
                # If the difference in vertical coordinates is too large, add curr_line to list of filtered lines and
                # set curr_line with the current line
                else:
                    new_line = Line((0, 0, 0, 0))
                    setattr(new_line, main_dim_1, min([getattr(l, main_dim_1) for l in sub_cluster]))
                    setattr(new_line, main_dim_2, max([getattr(l, main_dim_2) for l in sub_cluster]))
                    setattr(new_line, sec_dim_1, statistics.mean([getattr(l, sec_dim_1) for l in sub_cluster]))
                    setattr(new_line, sec_dim_2, statistics.mean([getattr(l, sec_dim_2) for l in sub_cluster]))
                    final_lines.append(new_line)
                    sub_cluster = [line]

        new_line = Line((0, 0, 0, 0))
        setattr(new_line, main_dim_1, min([getattr(l, main_dim_1) for l in sub_cluster]))
        setattr(new_line, main_dim_2, max([getattr(l, main_dim_2) for l in sub_cluster]))
        setattr(new_line, sec_dim_1, statistics.mean([getattr(l, sec_dim_1) for l in sub_cluster]))
        setattr(new_line, sec_dim_2, statistics.mean([getattr(l, sec_dim_2) for l in sub_cluster]))
        final_lines.append(new_line)

    # Remove "point" lines
    final_lines = [line for line in final_lines if not line.width == line.height == 0]

    return final_lines


def detect_lines(image: np.ndarray, rho: float = 1, theta: float = np.pi / 180, threshold: int = 50,
                 minLinLength: int = 290, maxLineGap: int = 6, classify: bool = True) -> (List[Line], List[Line]):
    """
    Detect horizontal and vertical lines on image
    :param image: image array
    :param rho: rho parameter for Hough line transform
    :param theta: theta parameter for Hough line transform
    :param threshold: threshold parameter for Hough line transform
    :param minLinLength: minLinLength parameter for Hough line transform
    :param maxLineGap: maxLineGap parameter for Hough line transform
    :param classify: boolean indicating if lines are classified into vertical and horizontal lines
    :return: horizontal and vertical lines
    """
    # Create copy of image
    img = image.copy()

    # Image to gray and canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(gray, 50, 200, None, 3)

    # Compute Hough lines on image
    linesP = cv2.HoughLinesP(dst, rho, theta, threshold, None, minLinLength, maxLineGap)

    # Parse lines to Line object
    lines = [Line(line=line[0]) for line in linesP]

    # If lines are not classified, return lines
    if not classify:
        return lines

    # Identify horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        # Reprocess line
        line.reprocess()

        if line.vertical:
            vertical_lines.append(line)
        elif line.horizontal:
            horizontal_lines.append(line)

    # Compute merged lines
    # horizontal_lines = overlapping_filter(lines=horizontal_lines, horizontal=True, max_gap=maxLineGap)
    # vertical_lines = overlapping_filter(lines=vertical_lines, horizontal=False, max_gap=maxLineGap)

    return horizontal_lines, vertical_lines
