
"""
Implementation of Adaptive RLSA algorithm based on https://www.sciencedirect.com/science/article/abs/pii/S0262885609002005
and text line segmentation by
"""

from typing import Optional

import cv2
import numpy as np
from numba import njit, prange

from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table


@njit("int32[:,:](int32[:,:],int32[:,:],float64,float64)", fastmath=True, cache=True, parallel=False)
def remove_noise(cc: np.ndarray, cc_stats: np.ndarray, average_height: float, median_width: float) -> np.ndarray:
    """
    Remove noise from detected connected components
    :param cc: connected components labels array
    :param cc_stats: connected components' statistics array
    :param average_height: average connected components' height
    :param median_width: median connected components' width
    :return: connected components labels array without noisy components
    """
    for idx in prange(len(cc_stats)):
        if idx == 0:
            continue

        # Get stats
        x, y, w, h, area = cc_stats[idx][:]

        # Check dashes
        is_dash = (w / h >= 2) and (0.5 * median_width <= w <= 1.5 * median_width)

        if is_dash:
            continue

        # Check removal conditions
        cond_height = h < average_height / 3
        cond_elongation = max(h, w) / max(min(h, w), 1) < 0.33
        cond_low_density = area / (max(w, 1) * max(h, 1)) < 0.08

        if cond_height or cond_elongation or cond_low_density:
            for row in range(y, y + h):
                for col in range(x, x + w):
                    if cc[row][col] == idx:
                        cc[row][col] = 0

    return cc


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
            if prev_cc_label == -1 or label == -1:
                prev_cc_position, prev_cc_label = col, label
                continue
            if label == prev_cc_label:
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
                for y in range(max(0, row - 2), min(row + 3, h)):
                    for x in range(prev_cc_position + 1, col):
                        if cc[y][x] not in list_ccs:
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


@njit("boolean[:,:](uint8[:,:],float64)", fastmath=True, cache=True, parallel=False)
def find_obstacles(img: np.ndarray, min_width: float) -> np.ndarray:
    """
    Identify obstacles (columns, line gaps) in image
    :param img: image array
    :param min_width: minimum width of obstacles
    :return: connected components labels array with obstacles identified
    """
    mask_obstacles = np.full(shape=img.shape, fill_value=False)
    min_width = int(np.ceil(min_width))
    h, w = img.shape

    for col in prange(w - min_width):
        prev_cc_position = -1
        for row in range(h):
            max_value = 0
            for idx in range(min_width):
                max_value = max(max_value, img[row][col + idx])

            # Not a CC
            if max_value == 0:
                continue

            length = row - prev_cc_position - 1
            if length > h / 5:
                for id_row in range(prev_cc_position + 1, row):
                    for idx in range(min_width):
                        mask_obstacles[id_row][col + idx] = True

            # Update counters
            prev_cc_position = row

        # Check ending
        length = row + 1 - prev_cc_position - 1
        if length > h / 5:
            for id_row in range(prev_cc_position + 1, row + 1):
                for idx in range(min_width):
                    mask_obstacles[id_row][col + idx] = True

    return mask_obstacles


@njit("boolean[:, :](uint8[:, :],int32[:, :],float64,float64)", fastmath=True, cache=True, parallel=False)
def get_text_mask(thresh: np.ndarray, cc_stats_rlsa: np.ndarray, char_length: float,
                  median_width: float) -> np.ndarray:
    """
    Identify image text mask
    :param thresh: thresholded image
    :param cc_stats_rlsa: connected components stats array
    :param char_length: average character length
    :param median_width: median connected components' width
    :return: text mask array
    """
    text_mask = np.full(shape=thresh.shape, fill_value=False)

    # Get average height
    num, denum = 0, 0
    for i in range(1, cc_stats_rlsa.shape[0]):
        height, area = cc_stats_rlsa[i, cv2.CC_STAT_HEIGHT], cc_stats_rlsa[i, cv2.CC_STAT_AREA]
        num += height * area
        denum += area
    Hm = num / max(denum, 1)

    for cc_idx in prange(len(cc_stats_rlsa)):
        x, y, w, h, area = cc_stats_rlsa[cc_idx][:]

        # Check for dashes
        if (w / h >= 2) and (0.5 * median_width <= w <= 1.5 * median_width):
            for row in prange(y, y + h):
                for col in prange(x, x + w):
                    text_mask[row][col] = True
            continue

        if cc_idx == 0 or min(w, h) <= 2 * char_length / 3:
            continue

        # Get horizontal white to black transitions
        h_tc = 0
        for row in prange(y, y + h):
            prev_value = 0
            for col in range(x, x + w):
                value = thresh[row][col]

                if value == 255 and prev_value == 0:
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
        if (0.8 * Hm <= H <= 1.2 * Hm) or ( 0.8 * Hm > H and 1.2 < THx < 3.5):
            is_text = True
        elif (THx < 0.2 and R > 5 and 0.95 < TVx < 1.05) or (THx > 5 and R < 0.2 and 0.95 < THy < 1.05):
            is_text = False
        elif 1.2 * Hm < H and 1.2 < THx < 3.5 and 1.2 < TVx < 3.5:
            is_text = True

        if is_text:
            for row in prange(y, y + h):
                for col in prange(x, x + w):
                    text_mask[row][col] = True

    return text_mask


def identify_text_mask(thresh: np.ndarray, lines: list[Line], char_length: float,
                       existing_tables: Optional[list[Table]] = None) -> np.ndarray:
    """
    Identify text mask of the input image
    :param thresh: threshold image array
    :param lines: list of image rows
    :param char_length: average character length
    :param existing_tables: list of detected bordered tables
    :return: thresholded image
    """
    # Mask rows in image
    for line in lines:
        if line.horizontal and line.length >= 3 * char_length:
            cv2.rectangle(thresh, (line.x1, line.y1 - line.thickness // 2 - 1), (line.x2, line.y2 + line.thickness // 2 + 1),
                          (0, 0, 0), -1)
        elif line.vertical and line.length >= 2 * char_length:
            cv2.rectangle(thresh, (line.x1 - line.thickness // 2 - 1, line.y1), (line.x2 + line.thickness // 2 + 1, line.y2),
                          (0, 0, 0), -1)

    # Apply dilation
    thresh = cv2.dilate(thresh, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)), iterations=1)

    # Connected components
    _, cc, cc_stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    if len(cc_stats) <= 1:
        return thresh

    # Remove noise
    average_height = np.mean(cc_stats[1:, cv2.CC_STAT_HEIGHT])
    median_width = np.median(cc_stats[1:, cv2.CC_STAT_WIDTH])
    cc_denoised = remove_noise(cc=cc, cc_stats=cc_stats, average_height=average_height, median_width=median_width)

    # Apply small RLSA
    rlsa_small = adaptive_rlsa(cc=cc_denoised, cc_stats=cc_stats, a=1, th=3.5, c=0.4)
    rlsa_small = cv2.erode(255 * (rlsa_small > 0).astype(np.uint8),
                           kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))

    # Identify obstacles and remove them from denoised cc array
    mask_obstacles = find_obstacles(img=np.maximum(rlsa_small, thresh),
                                    min_width=char_length)
    cc_obstacles = cc_denoised.copy()
    cc_obstacles[mask_obstacles] = -1

    # RLSA image
    rlsa_image = adaptive_rlsa(cc=cc_obstacles, cc_stats=cc_stats, a=5, th=3.5, c=0.4)

    # Connected components of the rlsa image
    _, _, cc_stats_rlsa, _ = cv2.connectedComponentsWithStats(255 * (rlsa_image > 0).astype(np.uint8), 8, cv2.CV_32S)

    # Get text mask
    text_mask = get_text_mask(thresh=thresh,
                              cc_stats_rlsa=cc_stats_rlsa,
                              char_length=char_length,
                              median_width=median_width)

    # Compute final image
    cc_final = cc_obstacles.copy()
    cc_final[~text_mask] = -1
    rlsa_final = adaptive_rlsa(cc=cc_final, cc_stats=cc_stats, a=1.25, th=3.5, c=0.4)

    # Remove all elements from existing tables
    for tb in existing_tables or []:
        rlsa_final[tb.y1:tb.y2, tb.x1:tb.x2] = 0

    return cv2.erode(255 * rlsa_final.astype(np.uint8),
                     kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
