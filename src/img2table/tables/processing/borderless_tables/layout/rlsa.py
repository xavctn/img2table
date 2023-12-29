# coding: utf-8

"""
Implementation of Adaptive RLSA algorithm based on https://www.sciencedirect.com/science/article/abs/pii/S0262885609002005
and text line segmentation by
"""

from typing import List

import cv2
import numpy as np
from numba import njit

from img2table.tables.objects.line import Line


def remove_noise(cc: np.ndarray, cc_stats: np.ndarray, average_height: int) -> np.ndarray:
    """
    Remove noise from detected connected components
    :param cc: connected components labels array
    :param cc_stats: connected components' statistics array
    :param average_height: average connected components' height
    :return: connected components labels array without noisy components
    """
    # Condition on height
    mask_height = cc_stats[:, cv2.CC_STAT_HEIGHT] < average_height / 3

    # Condition on elongation
    mask_elongation = (np.maximum(cc_stats[:, cv2.CC_STAT_HEIGHT], cc_stats[:, cv2.CC_STAT_WIDTH])
                       / np.minimum(cc_stats[:, cv2.CC_STAT_HEIGHT], cc_stats[:, cv2.CC_STAT_WIDTH])) < 0.08

    # Condition on low density
    mask_low_density = cc_stats[:, cv2.CC_STAT_AREA] / (cc_stats[:, cv2.CC_STAT_HEIGHT] * cc_stats[:, cv2.CC_STAT_WIDTH]) < 0.08

    # Create mask on noise
    mask_noise = mask_height | mask_elongation | mask_low_density

    # Create new connected components labels array without noisy components
    cc_denoised = cc.copy()
    cc_to_remove = [idx for idx, val in enumerate(mask_noise) if val]
    cc_denoised[np.isin(cc_denoised, cc_to_remove)] = 0

    return cc_denoised


@njit("uint8[:,:](int32[:,:],int32[:,:],float64,float64,float64)", fastmath=True)
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
    for row in range(h):
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
                height_ratio = max(height_cc, height_prev) / min(height_cc, height_prev)
                h_overlap = min(y1_cc + height_cc, y1_prev + height_prev) - max(y1_cc, y1_prev)

                # Presence of other CC
                no_other_cc = True
                list_ccs = [-1, 0, label, prev_cc_label]
                for y in range(max(0, row - 2), min(row + 3, h)):
                    for x in range(prev_cc_position + 1, col):
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


@njit("int32[:,:](int32[:,:],int32[:,:])", fastmath=True)
def find_obstacles(cc: np.ndarray, cc_rlsa: np.ndarray) -> np.ndarray:
    """
    Identify obstacles (columns, line gaps) in image
    :param cc: connected components labels array
    :param cc_rlsa: connected components labels array from RLSA image
    :return: connected components labels array with obstacles identified
    """
    h, w = cc.shape
    cc_obstacles = cc.copy()

    for col in range(w):
        prev_cc_position, prev_cc_label = -1, -1
        for row in range(h):
            label = cc_rlsa[row][col]

            # Not a CC
            if label == 0:
                continue
            else:
                if label != prev_cc_label:
                    length = row - prev_cc_position - 1
                    if length > h / 3:
                        for id_row in range(prev_cc_position + 1, row):
                            cc_obstacles[row][col] = -1

                # Update counters
                prev_cc_position, prev_cc_label = row, label

    return cc_obstacles


@njit("boolean[:, :](uint8[:, :],int32[:, :])", fastmath=True)
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

    for cc_idx in range(len(cc_stats_rlsa)):
        if cc_idx == 0:
            continue

        x, y, w, h, area = cc_stats_rlsa[cc_idx][:]

        # Get horizontal white to black transitions
        h_tc = 0
        for row in range(y, y + h):
            prev_value = 0
            for col in range(x, x + w):
                value = thresh[row][col]

                if value == 255:
                    if prev_value == 0:
                        h_tc += 1
                prev_value = value

        # Get vertical white to black transitions
        v_tc, nb_cols = 0, 0
        for col in range(x, x + w):
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
        H, R, THx, TVx, THy = h, w / h, h_tc / nb_cols, v_tc / nb_cols, h_tc / h

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
            for row in range(y, y + h):
                for col in range(x, x + w):
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

    # Apply first adaptive RLSA to CC
    rsla_step1 = adaptive_rlsa(cc=cc_denoised, cc_stats=cc_stats, a=1.5, th=3.5, c=0.4)

    # Get RLSA Connected Components
    _, cc_rlsa_1, cc_stats_rlsa_1, _ = cv2.connectedComponentsWithStats(255 * rsla_step1, 8, cv2.CV_32S)

    # Identify obstacles
    cc_obstacles = find_obstacles(cc=cc_denoised, cc_rlsa=cc_rlsa_1)

    # RLSA image
    rlsa_image = adaptive_rlsa(cc=cc_obstacles, cc_stats=cc_stats, a=5, th=3.5, c=0.4)

    # Connected components of the rlsa image
    _, _, cc_stats_rlsa, _ = cv2.connectedComponentsWithStats(255 * (rlsa_image > 0).astype(np.uint8), 8, cv2.CV_32S)

    text_mask = get_text_mask(thresh=thresh,
                              cc_stats_rlsa=cc_stats_rlsa)

    return text_mask
