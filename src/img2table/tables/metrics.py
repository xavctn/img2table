from typing import Optional

import cv2
import numpy as np
import polars as pl
from numba import njit, prange

from img2table.tables.objects.cell import Cell


@njit("int32[:,:](int32[:,:],int32[:,:])", fastmath=True, cache=True, parallel=False)
def remove_dots(cc_labels: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """
    Remove dots from connected components
    :param cc_labels: connected components' label array
    :param stats: connected components' stats array
    :return: list of non-dot connected components' indexes
    """
    cc_to_keep = []

    for idx in prange(len(stats)):
        if idx == 0:
            continue

        x, y, w, h, area = stats[idx][:]

        # Check number of inner pixels
        inner_pixels = 0
        for row in prange(y, y + h):
            prev_position = -1
            for col in range(x, x + w):
                value = cc_labels[row][col]
                if value == idx:
                    if prev_position >= 0:
                        inner_pixels += col - prev_position - 1
                    prev_position = col

        for col in prange(x, x + w):
            prev_position = -1
            for row in range(y, y + h):
                value = cc_labels[row][col]
                if value == idx:
                    if prev_position >= 0:
                        inner_pixels += row - prev_position - 1
                    prev_position = row

        # Compute roundness
        roundness = 4 * area / (np.pi * max(h, w) ** 2)

        if not (inner_pixels / (2 * area) <= 0.1 and roundness >= 0.7):
            cc_to_keep.append([x, y, w, h, area])

    return np.array(cc_to_keep) if cc_to_keep else np.empty((0, 5), dtype=np.int32)


@njit("int32[:,:](float64[:,:])", cache=True, fastmath=True, parallel=False)
def remove_dotted_lines(complete_stats: np.ndarray) -> np.ndarray:
    """
    Remove dotted lines in image by identifying aligned connected components
    :param complete_stats: connected components' array
    :return: filtered connected components' array
    """
    line_areas = []

    ### Identify horizontal lines
    complete_stats = complete_stats[complete_stats[:, 6].argsort()]

    x1_area, y1_area, x2_area, y2_area, width_area, prev_y_middle, area_count = 0, 0, 0, 0, 0, -10, 0
    for idx in prange(complete_stats.shape[0]):
        x, y, w, h, _, x_middle, y_middle = complete_stats[idx][:]

        if w / h < 2:
            continue

        if y_middle - prev_y_middle <= 2:
            # Add to previous area
            x1_area, y1_area, x2_area, y2_area = min(x, x1_area), min(y, y1_area), max(x + w, x2_area), max(y + h,
                                                                                                            y2_area)
            width_area += w
            area_count += 1
            prev_y_middle = y_middle
        else:
            # Check if previously defined area is relevant
            if area_count >= 5 and width_area / ((x2_area - x1_area) or 1) >= 0.66:
                line_areas.append([float(x1_area), float(y1_area), float(x2_area), float(y2_area)])
            # Create new area
            x1_area, y1_area, x2_area, y2_area = x, y, x + w, y + h
            width_area, prev_y_middle, area_count = w, y_middle, 1

    # Check last area
    if area_count >= 5 and width_area / ((x2_area - x1_area) or 1) >= 0.66:
        line_areas.append([float(x1_area), float(y1_area), float(x2_area), float(y2_area)])

    ### Identify vertical lines
    complete_stats = complete_stats[complete_stats[:, 5].argsort()]

    x1_area, y1_area, x2_area, y2_area, height_area, prev_x_middle, area_count = 0, 0, 0, 0, 0, -10, 0
    for idx in prange(complete_stats.shape[0]):
        x, y, w, h, _, x_middle, y_middle = complete_stats[idx][:]

        if h / w < 2:
            continue

        if x_middle - prev_x_middle <= 2:
            # Add to previous area
            x1_area, y1_area, x2_area, y2_area = min(x, x1_area), min(y, y1_area), max(x + w, x2_area), max(y + h,
                                                                                                            y2_area)
            height_area += h
            area_count += 1
            prev_x_middle = x_middle
        else:
            # Check if previously defined area is relevant
            if area_count >= 5 and height_area / ((y2_area - y1_area) or 1) >= 0.66:
                line_areas.append([float(x1_area), float(y1_area), float(x2_area), float(y2_area)])
            # Create new area
            x1_area, y1_area, x2_area, y2_area = x, y, x + w, y + h
            height_area, prev_x_middle, area_count = h, x_middle, 1

    # Check last area
    if area_count >= 5 and height_area / ((y2_area - y1_area) or 1) >= 0.66:
        line_areas.append([float(x1_area), float(y1_area), float(x2_area), float(y2_area)])

    if len(line_areas) == 0:
        return complete_stats[:, :5].astype(np.int32)

    # Create array of line areas
    areas_array = np.array(line_areas)

    # Check if connected components is located in areas
    kept_cc = []
    for idx in prange(complete_stats.shape[0]):
        x, y, w, h, area, x_middle, y_middle = complete_stats[idx][:]

        intersection_area = 0
        for j in range(areas_array.shape[0]):
            x1_area, y1_area, x2_area, y2_area = areas_array[j][:]

            # Compute overlaps
            x_overlap = max(0, min(x2_area, x + w) - max(x1_area, x))
            y_overlap = max(0, min(y2_area, y + h) - max(y1_area, y))
            intersection_area += x_overlap * y_overlap

        if intersection_area / (w * h) < 0.25:
            kept_cc.append([x, y, w, h, area])

    return np.array(kept_cc).astype(np.int32) if kept_cc else np.empty((0, 5), dtype=np.int32)


@njit("UniTuple(int32[:,:], 2)(int32[:,:])", cache=True, fastmath=True, parallel=False)
def filter_cc(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter relevant connected components
    :param stats: connected components' array
    :return: tuple with relevant connected components' array and discarded connected components' array
    """
    kept_cc, discarded_cc = [], []

    for idx in prange(stats.shape[0]):
        x, y, w, h, area = stats[idx][:]

        # Compute aspect ratio and fill ratio
        ar = max(w, h) / min(w, h)
        fill = area / (w * h)

        if ar <= 5 and fill > 0.08:
            kept_cc.append([x, y, w, h, area])
        else:
            discarded_cc.append([x, y, w, h, area])

    if len(kept_cc) == 0:
        # Map to arrays
        kept_array = np.array(kept_cc) if kept_cc else np.empty((0, 5), dtype=np.int32)
        discarded_array = np.array(discarded_cc) if discarded_cc else np.empty((0, 5), dtype=np.int32)
        return kept_array, discarded_array

    # Map kept_cc to array and compute metrics
    kept_stats = np.array(kept_cc)
    median_width = np.median(kept_stats[:, cv2.CC_STAT_WIDTH])
    median_height = np.median(kept_stats[:, cv2.CC_STAT_HEIGHT])

    # Compute bbox area bounds
    upper_bound = 5 * median_width * median_height
    lower_bound = 0.2 * median_width * median_height

    kept_cc = []
    for idx in prange(kept_stats.shape[0]):
        x, y, w, h, area = kept_stats[idx][:]

        # Check area
        bounded_area = lower_bound <= w * h <= upper_bound
        # Check dashes
        is_dash = (w / h >= 2) and (0.5 * median_width <= w <= 1.5 * median_width)

        if bounded_area or is_dash:
            kept_cc.append([x, y, w, h, area])
        else:
            discarded_cc.append([x, y, w, h, area])

    # Map to arrays
    kept_array = np.array(kept_cc) if kept_cc else np.empty((0, 5), dtype=np.int32)
    discarded_array = np.array(discarded_cc) if discarded_cc else np.empty((0, 5), dtype=np.int32)
    return kept_array, discarded_array


@njit("Tuple((uint8[:,:],int32[:,:]))(uint8[:,:],int32[:,:],int32[:,:],float64)", fastmath=True, cache=True,
      parallel=False)
def create_character_thresh(thresh: np.ndarray, stats: np.ndarray, discarded_stats: np.ndarray,
                            char_length: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create thresholded image containing uniquely characters
    :param thresh: thresholded image
    :param stats: relevant connected components' array
    :param discarded_stats: discarded connected components' array
    :param char_length: average character length
    :return: thresholded image containing uniquely characters and array of image characters
    """
    # Create blank character thresh
    character_thresh = np.zeros(thresh.shape, dtype=np.uint8)

    # Identify CC from discarded connected components that can be characters
    list_relevant_chars = []
    for idx in prange(len(stats)):
        x, y, w, h, area = stats[idx][:]

        # Add character to thresholded image
        list_relevant_chars.append([x, y, w, h, area])
        character_thresh[y:y + h, x:x + w] = thresh[y:y + h, x:x + w]

        for idx_discarded in prange(1, len(discarded_stats)):
            cc_x, cc_y, cc_w, cc_h, cc_area = discarded_stats[idx_discarded][:]

            # Compute y overlap
            y_overlap = min(cc_y + cc_h, y + h) - max(cc_y, y)

            if y_overlap < 0.5 * min(cc_h, h):
                continue
            if max(cc_h, cc_w) > 3 * max(h, w):
                continue

            # Compute horizontal distance
            distance = min(abs(cc_x - x), abs(cc_x - x - w), abs(cc_x + cc_w - x), abs(cc_x + cc_w - x - w))

            if y_overlap > 0 and distance <= char_length:
                # Add new character to thresholded image
                list_relevant_chars.append([cc_x, cc_y, cc_w, cc_h, cc_area])
                character_thresh[cc_y:cc_y + cc_h, cc_x:cc_x + cc_w] = thresh[cc_y:cc_y + cc_h, cc_x:cc_x + cc_w]

    return character_thresh, np.array(list_relevant_chars) if list_relevant_chars else np.empty((0, 5), dtype=np.int32)


def compute_char_length(thresh: np.ndarray) -> tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute average character length based on connected components' analysis
    :param thresh: threshold image array
    :return: tuple with average character length, thresholded image of characters and array of image characters
    """
    # Connected components
    _, cc_labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # Remove dots
    stats = remove_dots(cc_labels=cc_labels, stats=stats)

    # Remove connected components with less than 10 pixels
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 10
    stats = stats[mask_pixels]

    if len(stats) == 0:
        return None, None, None

    # Remove dotted lines
    complete_stats = np.c_[stats, (2 * stats[:, 0] + stats[:, 2]) / 2, (2 * stats[:, 1] + stats[:, 3]) / 2]
    stats = remove_dotted_lines(complete_stats=complete_stats)

    if len(stats) == 0:
        return None, None, None

    # Filter relevant connected components
    relevant_stats, discarded_stats = filter_cc(stats=stats)

    if len(relevant_stats) > 0:
        # Compute average character length
        argmax_char_length = float(np.argmax(np.bincount(relevant_stats[:, cv2.CC_STAT_WIDTH])))
        mean_char_length = np.mean(relevant_stats[:, cv2.CC_STAT_WIDTH])
        char_length = mean_char_length if 1.5 * argmax_char_length <= mean_char_length else argmax_char_length

        # Create thresholded image with characters
        characters_thresh, chars_array = create_character_thresh(thresh=thresh,
                                                                 stats=relevant_stats,
                                                                 discarded_stats=discarded_stats,
                                                                 char_length=char_length)

        return char_length, characters_thresh, chars_array
    return None, None, None


@njit("int64[:,:](int32[:,:],int32[:,:])", cache=True, fastmath=True)
def recompute_contours(stats: np.ndarray, chars_array: np.ndarray) -> np.ndarray:
    """
    Recompute contours from CC analysis with original characters
    :param stats: contours from CC analysis
    :param chars_array: characters array
    :return: array of contours with dimensions recomputed
    """
    list_contours = []
    for idx in prange(stats.shape[0]):
        if idx == 0:
            continue
        x, y, w, h, area = stats[idx][:]

        # Identify contour coordinates by matching included characters
        x1, y1, x2, y2, nb_chars = 10 ** 6, 10 ** 6, 0, 0, 0
        for id_c in range(chars_array.shape[0]):
            xc, yc, wc, hc, area = chars_array[id_c][:]

            # Compute overlaps
            x_overlap = max(0, min(x + w, xc + wc) - max(x, xc))
            y_overlap = max(0, min(y + h, yc + hc) - max(y, yc))

            if x_overlap * y_overlap >= 0.5 * hc * wc:
                # Update stats
                x1, y1, x2, y2 = min(x1, xc), min(y1, yc), max(x2, xc + wc), max(y2, yc + hc)
                nb_chars += 1

        if nb_chars > 0:
            list_contours.append([x1, y1, x2 - x1, y2 - y1])

    return np.array(list_contours) if list_contours else np.empty((0, 4), dtype=np.int64)


@njit("List(float64)(int64[:,:],float64)", cache=True, fastmath=True, parallel=False)
def get_row_separations(stats: np.ndarray, char_length: float) -> list[float]:
    """
    Compute row separation between contours
    :param stats: array of contours
    :param char_length: average character length
    :return: list of row separations
    """
    row_separations = []

    for i in prange(len(stats)):
        # Get statistics
        xi, yi, wi, hi = stats[i][:]
        row_separation = 10 ** 6

        for j in range(len(stats)):
            if i == j:
                continue

            # Get statistics
            xj, yj, wj, hj = stats[j][:]

            # Compute horizontal overlap and vertical positions
            h_overlap = min(xi + hi, xj + hj) - max(xi, xj)
            v_pos_i, v_pos_j = (2 * yi + hi) / 2, (2 * yj + hj) / 2
            if h_overlap <= char_length // 2 or v_pos_j <= v_pos_i:
                continue

            row_separation = min(row_separation, v_pos_j - v_pos_i)

        if row_separation < 10 ** 6:
            row_separations.append(row_separation)

    return row_separations


def compute_median_line_sep(thresh_chars: np.ndarray, chars_array: np.ndarray,
                            char_length: float) -> tuple[Optional[float], Optional[list[Cell]]]:
    """
    Compute median separation between rows
    :param thresh_chars: thresholded image of characters
    :param char_length: average character length
    :return: median separation between rows
    """
    # Identify characters that belong to the same word and create merged contours, by closing image and retrieving
    # connected components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(char_length // 2 + 1), int(char_length // 3 + 1)))
    thresh_chars = cv2.morphologyEx(thresh_chars, cv2.MORPH_CLOSE, kernel)

    _, _, stats, _ = cv2.connectedComponentsWithStats(thresh_chars, 8, cv2.CV_32S)

    # Recompute contours
    stats_contours = recompute_contours(stats=stats, chars_array=chars_array)

    # Compute median line sep
    row_separations = get_row_separations(stats=stats_contours, char_length=char_length)

    if row_separations:
        median_line_sep = (pl.DataFrame(row_separations, schema={"sep": float})
                           .with_columns(sep=2 * pl.col("sep").floordiv(2) + 1)
                           .group_by("sep").len().sort(by=['len', 'sep'], descending=[True, False])
                           .limit(1).to_dicts().pop().get("sep")
                           )
    else:
        median_line_sep = None

    # Get contours cells
    contours_cells = [Cell(x1=x, y1=y, x2=x + w, y2=y + h)
                      for idx, (x, y, w, h) in enumerate(stats_contours)
                      if idx > 0]

    return median_line_sep, contours_cells


def compute_img_metrics(thresh: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[list[Cell]]]:
    """
    Compute metrics from image
    :param thresh: threshold image array
    :return: average character length, median line separation and image contours
    """
    # Compute average character length based on connected components analysis
    char_length, thresh_chars, chars_array = compute_char_length(thresh=thresh)

    if char_length is None:
        return None, None, None

    # Compute median separation between rows
    median_line_sep, contours = compute_median_line_sep(thresh_chars=thresh_chars,
                                                        chars_array=chars_array,
                                                        char_length=char_length)

    return char_length, median_line_sep, contours
