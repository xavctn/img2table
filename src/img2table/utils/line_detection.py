# coding: utf-8
import statistics
from typing import List

import numpy as np
from cv2 import cv2

from img2table.objects.tables import Line


def horizontal_overlapping_filter(lines: List[Line]) -> List[Line]:
    """
    Process horizontal lines to merge close lines
    :param lines: horizontal lines
    :return: list of filtered horizontal lines
    """
    if len(lines) == 0:
        return []

    # Instantiate list of filtered lines
    filtered_lines = []

    # Sort lines by vertical position
    lines = sorted(lines, key=lambda l: (l.y1, l.x1))

    # Loop over lines to merge relevant lines together
    for idx, line in enumerate(lines):
        if idx == 0:
            curr_line = line
            curr_list_lines = [line]
        else:
            # Compute vertical difference between consecutive lines
            diff_index = line.y1 - curr_line.y1
            # If the difference is too large, add curr_line to list of filtered lines and set curr_line with the current
            # line
            if diff_index > 10:
                # Create average
                curr_line.y1 = curr_line.y2 = statistics.mean([l.y1 for l in curr_list_lines])
                filtered_lines.append(curr_line)
                curr_line = line
                curr_list_lines = [line]
            # If the difference is small enough, update curr_line based on current line coordinated
            else:
                curr_line.x1 = min(curr_line.x1, line.x1)
                curr_line.x2 = max(curr_line.x2, line.x2)
                curr_list_lines += [line]

    curr_line.y1 = curr_line.y2 = statistics.mean([l.y1 for l in curr_list_lines])
    filtered_lines.append(curr_line)

    return filtered_lines


def vertical_overlapping_filter(lines: List[Line]) -> List[Line]:
    """
    Process vertical lines to merge close lines
    :param lines: vertical lines
    :return: list of filtered vertical lines
    """
    if len(lines) == 0:
        return []

    # Sort lines by horizontal position
    lines = sorted(lines, key=lambda l: (l.x1, l.y1))

    # Create clusters of lines based on "similar" horizontal position
    line_clusters = list()
    for idx, line in enumerate(lines):
        if idx == 0:
            curr_cluster = [line]
        else:
            # Compute vertical difference between consecutive lines
            diff_index = line.x1 - curr_cluster[-1].x1
            # If the difference is too large, add curr_cluster to list clusters and set new cluster with the current
            # line
            if diff_index > 10:
                line_clusters.append(curr_cluster)
                curr_cluster = [line]
            # Otherwise, set line coordinates to coherent cluster values and append line to current cluster
            else:
                line.x1 = curr_cluster[0].x1
                line.x2 = curr_cluster[0].x2
                curr_cluster += [line]
    line_clusters.append(curr_cluster)

    # Create final lines by "merging" lines within a cluster
    final_lines = list()
    for cluster in line_clusters:
        # Sort the cluster
        cluster = sorted(cluster, key=lambda l: min(l.y1, l.y2))

        # Loop over lines in the cluster to merge relevant lines together
        for idx, line in enumerate(cluster):
            if idx == 0:
                curr_line = line
            else:
                # If lines are vertically close, merge line with curr_line
                if line.y1 <= curr_line.y2 + 20:
                    curr_line.y1 = min(curr_line.y1, line.y1)
                    curr_line.y2 = max(curr_line.y2, line.y2)
                # If the difference in vertical coordinates is too large, add curr_line to list of filtered lines and
                # set curr_line with the current line
                else:
                    final_lines.append(curr_line)
                    curr_line = line

        final_lines.append(curr_line)

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
    horizontal_lines = horizontal_overlapping_filter(lines=horizontal_lines)
    vertical_lines = vertical_overlapping_filter(lines=vertical_lines)

    return horizontal_lines, vertical_lines
