"""
ARLSA pipeline based on:
Nikolaou et al., "A segmentation framework for historical machine-printed documents",
Image and Vision Computing 28 (2010) 590-604.
"""

import cv2
import numpy as np
from numba import njit

from img2table.tables import find_components
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table


def _mask_detected_lines(
    thresh: np.ndarray, lines: list[Line] | None, char_length: float | None
) -> np.ndarray:
    """
    Remove lines from the thresholded image.
    :param thresh: threshold image array
    :param lines: list of image rows
    :param char_length: average character length
    :return: thresholded image with lines masked out
    """
    if lines is None or char_length is None:
        return thresh

    for line in lines:
        if line.horizontal and line.length >= 3 * char_length:
            cv2.rectangle(
                thresh,
                (line.x1, line.y1 - line.thickness // 2 - 1),
                (line.x2, line.y2 + line.thickness // 2 + 1),
                0,
                -1,
            )
        elif line.vertical and line.length >= 2 * char_length:
            cv2.rectangle(
                thresh,
                (line.x1 - line.thickness // 2 - 1, line.y1),
                (line.x2 + line.thickness // 2 + 1, line.y2),
                0,
                -1,
            )

    return thresh


@njit("boolean(int32[:,:],int32[:,:],int64)", fastmath=True, cache=True, parallel=False)
def _is_dot_component(cc: np.ndarray, cc_stats: np.ndarray, idx: int) -> bool:
    """
    Identify round dot-like connected components.
    :param cc: connected components labels array
    :param cc_stats: connected components' statistics array
    :param idx: connected component index
    :return: flag indicating whether the component is dot-like
    """
    x_cc, y_cc, w_cc, h_cc, area = cc_stats[idx][:5]

    if area <= 0:
        return False

    inner_pixels = 0
    for row in range(y_cc, y_cc + h_cc):
        prev_position = -1
        for col in range(x_cc, x_cc + w_cc):
            if cc[row][col] == idx:
                if prev_position >= 0:
                    inner_pixels += col - prev_position - 1
                prev_position = col

    for col in range(x_cc, x_cc + w_cc):
        prev_position = -1
        for row in range(y_cc, y_cc + h_cc):
            if cc[row][col] == idx:
                if prev_position >= 0:
                    inner_pixels += row - prev_position - 1
                prev_position = row

    roundness = 4.0 * area / (np.pi * max(h_cc, w_cc) ** 2)
    return (inner_pixels / (2.0 * area) <= 0.1) and (roundness >= 0.7)


@njit(
    "int32[:,:](int32[:,:],int32[:,:],float64,float64)", fastmath=True, cache=True, parallel=False
)
def remove_noise(
    cc: np.ndarray, cc_stats: np.ndarray, average_height: float, median_width: float
) -> np.ndarray:
    """
    Remove noise from detected connected components
    :param cc: connected components labels array
    :param cc_stats: connected components' statistics array
    :param average_height: average connected components' height
    :param median_width: median connected components' width
    :return: connected components labels array without noisy components
    """
    # Create lookup table of connected components labels to keep
    keep_label = np.ones(len(cc_stats), dtype=np.bool_)
    keep_label[0] = False

    for idx in range(1, len(cc_stats)):
        _, _, w_cc, h_cc, area = cc_stats[idx][:5]

        # Check dashes
        is_dash = (w_cc / h_cc >= 2) and (0.5 * median_width <= w_cc <= 1.5 * median_width)
        is_dot = _is_dot_component(cc=cc, cc_stats=cc_stats, idx=idx)
        if is_dash or is_dot:
            continue

        # Metrics
        elongation = min(h_cc, w_cc) / max(h_cc, w_cc, 1)
        density = area / (max(w_cc, 1) * max(h_cc, 1))

        # Mark for removal
        if h_cc < (average_height / 3.0) or density < 0.08 or elongation < 0.08:
            keep_label[idx] = False

    for row in range(cc.shape[0]):
        for col in range(cc.shape[1]):
            idx = cc[row, col]
            if idx > 0 and not keep_label[idx]:
                cc[row, col] = 0

    return cc


@njit(
    "uint8[:,:](int32[:,:],int32[:,:],uint8[:,:],float64,float64,float64)",
    fastmath=True,
    cache=True,
    parallel=False,
)
def adaptive_rlsa(
    cc: np.ndarray,
    cc_stats: np.ndarray,
    obstacle_mask: np.ndarray,
    a: float,
    th: float,
    c: float,
) -> np.ndarray:
    """
    Implementation of adaptive run-length smoothing algorithm with obstacle blocking
    :param cc: connected components labels array
    :param cc_stats: connected components' statistics array
    :param obstacle_mask: obstacle mask array
    :param a: connected components' distance ratio
    :param th: connected components' height ratio
    :param c: connected components' vertical overlap
    :return: RLSA resulting image
    """
    rlsa_img = (cc > 0).astype(np.uint8)
    h, w = cc.shape

    for row in range(h):
        prev_cc_position = -1
        prev_cc_label = 0

        for col in range(w):
            label = cc[row][col]
            if label == 0:
                continue

            if prev_cc_label == 0:
                prev_cc_position = col
                prev_cc_label = label
                continue

            # Pixels between two points of the same component are always filled.
            if label == prev_cc_label:
                rlsa_img[row][prev_cc_position:col] = 1
                prev_cc_position = col
                prev_cc_label = label
                continue

            length = col - prev_cc_position - 1
            if length <= 0:
                prev_cc_position = col
                prev_cc_label = label
                continue

            # Check for cross of obstacle pixels
            crosses_obstacle = False
            for x in range(prev_cc_position + 1, col):
                if obstacle_mask[row][x] > 0:
                    crosses_obstacle = True
                    break
            if crosses_obstacle:
                prev_cc_position = col
                prev_cc_label = label
                continue

            # Geometric constraints
            _x1_cc, y1_cc, _w_cc, h_cc = cc_stats[label][:4]
            _x1_prev, y1_prev, _w_prev, h_prev = cc_stats[prev_cc_label][:4]
            height_ratio = max(h_cc, h_prev) / max(min(h_cc, h_prev), 1)
            h_overlap = min(y1_cc + h_cc, y1_prev + h_prev) - max(y1_cc, y1_prev)

            # Check for other CC in the 3x3 neighborhood of sequence pixels.
            no_other_cc = True
            for x in range(prev_cc_position + 1, col):
                for ny in range(max(0, row - 1), min(h, row + 2)):
                    for nx in range(max(0, x - 1), min(w, x + 2)):
                        cc_value = cc[ny][nx]
                        if cc_value != 0 and cc_value != label and cc_value != prev_cc_label:  # noqa: PLR1714
                            no_other_cc = False
                            break
                    if not no_other_cc:
                        break
                if not no_other_cc:
                    break

            if (
                length <= a * min(h_cc, h_prev)
                and height_ratio <= th
                and h_overlap >= c * min(h_cc, h_prev)
                and no_other_cc
            ):
                rlsa_img[row][prev_cc_position:col] = 1

            prev_cc_position = col
            prev_cc_label = label

    return rlsa_img


def _apply_arlsa(img: np.ndarray, a: float, obstacle_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Apply ARLSA to a binary image and return a binary image.
    :param img: input binary image
    :param a: connected components' distance ratio
    :param obstacle_mask: obstacle mask array
    :return: RLSA resulting image
    """
    if obstacle_mask is None:
        obstacle_mask = np.zeros(shape=img.shape, dtype=np.uint8)

    _, cc, cc_stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)
    if len(cc_stats) <= 1:
        return img.copy()

    rlsa = adaptive_rlsa(
        cc=cc,
        cc_stats=cc_stats,
        obstacle_mask=obstacle_mask,
        a=float(a),
        th=3.5,
        c=0.4,
    )
    return (rlsa > 0).astype(np.uint8) * 255


@njit(
    "Tuple((uint8[:,:], uint8[:,:]))(uint8[:,:], int32[:,:], int32[:,:])",
    fastmath=True,
    cache=True,
    parallel=False,
)
def _remove_punctuation_marks(
    thresh: np.ndarray, labels: np.ndarray, stats: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create images with punctuation marks removed as well as image of punctuation marks
    :param thresh: input threshold image
    :param labels: connected components labels
    :param stats: connected components stats
    """
    # Create cleaned and punctuation images
    cleaned = np.zeros(shape=thresh.shape, dtype=np.uint8)
    punctuation = np.zeros(shape=thresh.shape, dtype=np.uint8)

    # Compute number of non zeros elements by labels
    non_zeros = np.zeros(shape=(stats.shape[0]), dtype=np.uint16)
    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            cc_idx = labels[row, col]
            if cc_idx > 0 and thresh[row, col] > 0:
                non_zeros[cc_idx] += 1

    # Identify punctuation connected components
    is_punct_map = np.zeros(stats.shape[0], dtype=np.bool_)
    for cc_idx in range(1, stats.shape[0]):
        area = stats[cc_idx, 4]

        # Assess if element is punctuation
        if non_zeros[cc_idx] > 0 and area / non_zeros[cc_idx] <= 1.15:
            is_punct_map[cc_idx] = True

    # Add to punctuation or cleaned image
    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            cc_idx = labels[row, col]
            if cc_idx > 0:
                val = thresh[row, col]
                if is_punct_map[cc_idx]:
                    punctuation[row, col] = val
                else:
                    cleaned[row, col] = val

    return cleaned, punctuation


def remove_punctuation_marks(thresh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove punctuation marks from a binary image using ARLSA
    :param thresh: input threshold image
    :return: input image with punctuation removee
    """
    # Apply RLSA
    rlsa = _apply_arlsa(img=thresh.copy(), a=1.5)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(rlsa, connectivity=8, ltype=cv2.CV_32S)

    # Create cleaned and punctuation images
    return _remove_punctuation_marks(thresh=thresh, labels=labels, stats=stats)


@njit("uint8[:,:](uint8[:,:],float64)", fastmath=True, cache=True, parallel=False)
def detect_obstacles(img: np.ndarray, min_width: float) -> np.ndarray:
    """
    Identify obstacles (columns, line gaps) in image
    :param img: image array
    :param min_width: minimum width of obstacles
    :return: connected components labels array with obstacles identified
    """
    mask_obstacles = np.zeros(shape=img.shape, dtype=np.uint8)
    min_width = max(1, int(np.ceil(min_width)))
    h, w = img.shape

    for col in range(w - min_width + 1):
        prev_cc_position = -1
        for row in range(h):
            max_value = 0
            for idx in range(min_width):
                max_value = max(max_value, img[row][col + idx])

            # Not a CC
            if max_value == 0:
                continue

            length = row - prev_cc_position - 1
            if length > h / 20:
                for id_row in range(prev_cc_position + 1, row):
                    for idx in range(min_width):
                        mask_obstacles[id_row][col + idx] = 1

            # Update counters
            prev_cc_position = row

        # Check ending
        length = row + 1 - prev_cc_position - 1
        if length > h / 20:
            for id_row in range(prev_cc_position + 1, row + 1):
                for idx in range(min_width):
                    mask_obstacles[id_row][col + idx] = 1

    return mask_obstacles


def identify_text_mask(
    thresh: np.ndarray,
    lines: list[Line],
    char_length: float,
    existing_tables: list[Table] | None = None,
) -> np.ndarray:
    """
    Identify text mask of the input image
    :param thresh: threshold image array
    :param lines: list of image rows
    :param char_length: average character length
    :param existing_tables: list of detected bordered tables
    :return: thresholded image corresponding to the text mask
    """
    # Remove lines from thresholded image
    thresh = _mask_detected_lines(thresh=thresh, lines=lines, char_length=char_length)

    # Compute connected components
    _, cc, cc_stats, _ = cv2.connectedComponentsWithStats(
        image=thresh, connectivity=8, ltype=cv2.CV_32S
    )
    if len(cc_stats) <= 1:
        return thresh

    average_height = float(np.mean(cc_stats[1:, cv2.CC_STAT_HEIGHT]))
    median_width = float(np.median(cc_stats[1:, cv2.CC_STAT_WIDTH]))
    cc_denoised = remove_noise(
        cc=cc.copy(), cc_stats=cc_stats, average_height=average_height, median_width=median_width
    )
    denoised = (cc_denoised > 0).astype(np.uint8) * 255

    # Remove punctuation
    cleaned, punctuation = remove_punctuation_marks(thresh=denoised)

    # Stage 3: obstacle detection on ARLSA(a=1.5) result
    obstacle_input = _apply_arlsa(cleaned, a=0.75)
    obstacles = detect_obstacles(img=obstacle_input, min_width=char_length)

    # Identify text lines and put back punctuation in final mask
    text_lines = _apply_arlsa(img=cleaned, a=4.0, obstacle_mask=obstacles)
    final_mask = np.maximum(text_lines, punctuation)

    # Remove elements from existing table positions
    for tb in existing_tables or []:
        final_mask[tb.y1 : tb.y2, tb.x1 : tb.x2] = 0

    return final_mask


def regroup_contours(cnts: list[Cell], char_length: float) -> list[Cell]:
    """
    Merge close contours together
    :param cnts: list of contours
    :param char_length: average character length
    :return: list of grouped contours
    """
    cnts = sorted(cnts, key=lambda cnt: (cnt.y1, cnt.x1))

    # Identify matching contours
    edges = []
    for i, cnt1 in enumerate(cnts):
        for j in range(i, len(cnts)):
            cnt2 = cnts[j]

            if cnt2.y1 >= cnt1.y2:
                break

            # Compute y overlap and check correspondence
            y_overlap = min(cnt1.y2, cnt2.y2) - max(cnt1.y1, cnt2.y1)
            if y_overlap < 0.25 * min(cnt1.height, cnt2.height):
                continue

            # Compute x overlap / distance and check correspondence
            x_overlap = min(cnt1.x2, cnt2.x2) - max(cnt1.x1, cnt2.x1)
            if x_overlap < -char_length:
                continue

            edges.append({i, j})

    # Identify groups
    contour_groups = find_components(edges=edges)

    return [
        Cell(
            x1=min(cnts[idx].x1 for idx in gp),
            y1=min(cnts[idx].y1 for idx in gp),
            x2=max(cnts[idx].x2 for idx in gp),
            y2=max(cnts[idx].y2 for idx in gp),
        )
        for gp in contour_groups
    ]


def identify_image_contours(
    thresh: np.ndarray,
    lines: list[Line],
    char_length: float,
    existing_tables: list[Table] | None = None,
) -> list[Cell]:
    """
    Identify contours of the input image
    :param thresh: threshold image array
    :param lines: list of image rows
    :param char_length: average character length
    :param existing_tables: list of detected bordered tables
    :return: list of contours identified as cells
    """
    # Get text mask
    text_mask = identify_text_mask(
        thresh=thresh, lines=lines, char_length=char_length, existing_tables=existing_tables
    )

    # Find contours, highlight text areas, and extract ROIs
    cnts, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Regroup raw contours first so nearby dots can merge into their text contour.
    contours = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        contours.append(Cell(x1=x, y1=y, x2=x + w, y2=y + h))

    grouped_contours = regroup_contours(cnts=contours, char_length=char_length)

    return [
        cnt
        for cnt in grouped_contours
        if (min(cnt.height, cnt.width) >= 0.5 * char_length and max(cnt.height, cnt.width) >= char_length)
        or (cnt.width / cnt.height >= 2 and 0.5 * char_length <= cnt.width <= 1.5 * char_length)
    ]
