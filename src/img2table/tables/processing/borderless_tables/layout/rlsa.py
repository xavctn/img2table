# coding: utf-8

"""
Implementation of Adaptive RLSA algorithm based on https://www.sciencedirect.com/science/article/abs/pii/S0262885609002005
and text line segmentation by
"""

from typing import List

import cv2
import numpy as np
from numba import njit, prange

from img2table.tables.objects.line import Line


@njit("int32[:,:](int32[:,:],int32[:,:],float64)", fastmath=True, cache=True, parallel=False)
def remove_noise(cc: np.ndarray, cc_stats: np.ndarray, average_height: float) -> np.ndarray:
    """
    Remove noise from detected connected components
    :param cc: connected components labels array
    :param cc_stats: connected components' statistics array
    :param average_height: average connected components' height
    :return: connected components labels array without noisy components
    """
    cc_denoised = cc.copy()
    for idx in prange(len(cc_stats)):
        if idx == 0:
            continue

        # Get stats
        x, y, w, h, area = cc_stats[idx][:]

        # Check removal conditions
        cond_height = h < average_height / 3
        cond_elongation = max(h, w) / max(min(h, w), 1) < 0.08
        cond_low_density = area / (max(w, 1) * max(h, 1)) < 0.08

        if cond_height or cond_elongation or cond_low_density:
            for row in prange(y, y + h):
                for col in prange(x, x + w):
                    if cc_denoised[row][col] == idx:
                        cc_denoised[row][col] = 0

    return cc_denoised


@njit("uint8[:,:](int32[:,:],int32[:,:],float64,float64,float64)", fastmath=True, cache=True, parallel=False)
def adaptive_rlsa(cc: np.ndarray, cc_stats: np.ndarray, a: float, th: float, c: float) -> np.ndarray:
    """
    Implementation of adaptive run-length smoothing algorithm
    :param cc: connected components labels array
    :param cc_stats: connected components' statistics array
    :param a: connected components' distance ratio
    :param th: connected components' height ratio
    :param c: connected components' vertical overlap
    :return: RLSA resulting image
    """
    rsla_img = (cc > 0).astype(np.uint8)

    h, w = cc.shape
    for row in prange(h):
        prev_cc_position, prev_cc_label = -1, -1
        for col in range(w):
            label = cc[row][col]

            # Not a CC
            if label == 0:
                continue
            # First encountered CC
            elif prev_cc_label == -1 or label == -1:
                prev_cc_position, prev_cc_label = col, label
                continue
            elif label == prev_cc_label:
                # Update all pixels in range
                rsla_img[row][prev_cc_position:col] = 1
            else:
                # Get CC characteristics
                x1_cc, y1_cc, width_cc, height_cc = cc_stats[label][:4]

                # Get other CC characteristics
                x1_prev, y1_prev, width_prev, height_prev = cc_stats[prev_cc_label][:4]

                # Compute metrics
                length = col - prev_cc_position - 1
                height_ratio = max(height_cc, height_prev) / max(min(height_cc, height_prev), 1)
                h_overlap = min(y1_cc + height_cc, y1_prev + height_prev) - max(y1_cc, y1_prev)

                # Presence of other CC
                no_other_cc = True
                list_ccs = [-1, 0, label, prev_cc_label]
                for y in prange(max(0, row - 2), min(row + 3, h)):
                    for x in prange(prev_cc_position + 1, col):
                        if not cc[y][x] in list_ccs:
                            no_other_cc = False

                # Check conditions
                if ((length <= a * min(height_cc, height_prev))
                        and (height_ratio <= th)
                        and (h_overlap >= c * min(height_cc, height_prev))
                        and no_other_cc
                ):
                    rsla_img[row][prev_cc_position:col] = 1

            # Update counters
            prev_cc_position, prev_cc_label = col, label

    return rsla_img


@njit("int32[:,:](int32[:,:])", fastmath=True, cache=True, parallel=False)
def find_obstacles(cc: np.ndarray) -> np.ndarray:
    """
    Identify obstacles (columns, line gaps) in image
    :param cc: connected components labels array
    :return: connected components labels array with obstacles identified
    """
    h, w = cc.shape
    cc_obstacles = cc.copy()

    for col in prange(w):
        prev_cc_position, prev_cc_label = -1, -1
        for row in range(h):
            label = cc_obstacles[row][col]

            # Not a CC
            if label == 0:
                continue
            else:
                if label != prev_cc_label:
                    length = row - prev_cc_position - 1
                    if length > h / 3:
                        for id_row in prange(prev_cc_position + 1, row):
                            cc_obstacles[id_row][col] = -1

                # Update counters
                prev_cc_position, prev_cc_label = row, label

    return cc_obstacles


@njit("boolean[:, :](uint8[:, :],int32[:, :])", fastmath=True, cache=True, parallel=False)
def get_text_mask(thresh: np.ndarray, cc_stats_rlsa: np.ndarray) -> np.ndarray:
    """
    Identify image text mask
    :param thresh: thresholded image
    :param cc_stats_rlsa: connected components stats array
    :return: text mask array
    """
    text_mask = np.full(shape=thresh.shape, fill_value=False)

    # Get average height
    Hm = np.mean(cc_stats_rlsa[1:, cv2.CC_STAT_HEIGHT])

    for cc_idx in prange(len(cc_stats_rlsa)):
        if cc_idx == 0:
            continue

        x, y, w, h, area = cc_stats_rlsa[cc_idx][:]

        # Get horizontal white to black transitions
        h_tc = 0
        for row in prange(y, y + h):
            prev_value = 0
            for col in range(x, x + w):
                value = thresh[row][col]

                if value == 255:
                    if prev_value == 0:
                        h_tc += 1
                prev_value = value

        # Get vertical white to black transitions
        v_tc, nb_cols = 0, 0
        for col in prange(x, x + w):
            has_pixel, prev_value = 0, 0
            for row in range(y, y + h):
                value = thresh[row][col]

                if value == 255:
                    has_pixel = 1
                    if prev_value == 0:
                        v_tc += 1
                prev_value = value

            nb_cols += has_pixel

        # Update metrics
        H, R, THx, TVx, THy = h, w / max(h, 1), h_tc / max(nb_cols, 1), v_tc / max(nb_cols, 1), h_tc / max(h, 1)

        # Apply rules to identify text elements
        is_text = False
        if 0.8 * Hm <= H <= 1.2 * Hm:
            is_text = True
        elif H < 0.8 * Hm and 1.2 < THx < 3.0:
            is_text = True
        elif THx < 0.2 and R > 5 and 0.95 < TVx < 1.05:
            is_text = False
        elif THx > 5 and R < 0.2 and 0.95 < THy < 1.05:
            is_text = False
        elif H > 1.2 * Hm and 1.2 < THx < 3.0 and 1.2 < TVx < 3.5:
            is_text = True

        if is_text:
            for row in prange(y, y + h):
                for col in prange(x, x + w):
                    text_mask[row][col] = True

    return text_mask


def identify_text_mask(thresh: np.ndarray, lines: List[Line], char_length: float) -> np.ndarray:
    """
    Identify text mask of the input image
    :param thresh: thresholded image array
    :param lines: list of image rows
    :param char_length: average character length
    :return: thresholded image and text mask array
    """
    # Mask rows
    for l in lines:
        if l.horizontal and l.length >= 3 * char_length:
            cv2.rectangle(thresh, (l.x1 - l.thickness, l.y1), (l.x2 + l.thickness, l.y2), (0, 0, 0), 3 * l.thickness)
        elif l.vertical and l.length >= 2 * char_length:
            cv2.rectangle(thresh, (l.x1, l.y1 - l.thickness), (l.x2, l.y2 + l.thickness), (0, 0, 0), 3 * l.thickness)

    # Connected components
    _, cc, cc_stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    if len(cc_stats) <= 1:
        return thresh

    # Remove noise
    cc_denoised = remove_noise(cc=cc, cc_stats=cc_stats, average_height=char_length)

    # Identify obstacles
    cc_obstacles = find_obstacles(cc=cc_denoised)

    # RLSA image
    rlsa_image = adaptive_rlsa(cc=cc_obstacles, cc_stats=cc_stats, a=5, th=3.5, c=0.4)

    # Connected components of the rlsa image
    _, _, cc_stats_rlsa, _ = cv2.connectedComponentsWithStats(255 * (rlsa_image > 0).astype(np.uint8), 8, cv2.CV_32S)

    # Get text mask
    text_mask = get_text_mask(thresh=thresh,
                              cc_stats_rlsa=cc_stats_rlsa)

    # Filter thresholded image with the text mask
    text_thresh = thresh.copy()
    text_thresh[~text_mask] = 0

    return text_thresh
