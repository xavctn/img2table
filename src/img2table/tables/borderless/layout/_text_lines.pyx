cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport M_PI, ceil

cnp.import_array()


cdef inline double _max_double(double a, double b) noexcept:
    return a if a >= b else b


cdef inline double _min_double(double a, double b) noexcept:
    return a if a <= b else b


cdef inline long _max_long(long a, long b) noexcept:
    return a if a >= b else b


cdef inline long _min_long(long a, long b) noexcept:
    return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _is_dot_component(
    cnp.ndarray[cnp.int32_t, ndim=2] cc,
    cnp.ndarray[cnp.int32_t, ndim=2] cc_stats,
    long idx,
):
    """
    Identify round dot-like connected components.
    :param cc: connected components labels array
    :param cc_stats: connected components' statistics array
    :param idx: connected component index
    :return: flag indicating whether the component is dot-like
    """
    cdef cnp.int32_t[:, ::1] cc_view = cc
    cdef cnp.int32_t[:, ::1] stats_view = cc_stats
    cdef long x_cc, y_cc, w_cc, h_cc, area
    cdef long inner_pixels, prev_position
    cdef Py_ssize_t row, col
    cdef double roundness

    x_cc = stats_view[idx, 0]
    y_cc = stats_view[idx, 1]
    w_cc = stats_view[idx, 2]
    h_cc = stats_view[idx, 3]
    area = stats_view[idx, 4]

    if area <= 0:
        return False

    inner_pixels = 0
    for row in range(y_cc, y_cc + h_cc):
        prev_position = -1
        for col in range(x_cc, x_cc + w_cc):
            if cc_view[row, col] == idx:
                if prev_position >= 0:
                    inner_pixels += col - prev_position - 1
                prev_position = col

    for col in range(x_cc, x_cc + w_cc):
        prev_position = -1
        for row in range(y_cc, y_cc + h_cc):
            if cc_view[row, col] == idx:
                if prev_position >= 0:
                    inner_pixels += row - prev_position - 1
                prev_position = row

    roundness = 4.0 * area / (M_PI * _max_long(h_cc, w_cc) ** 2)
    return (inner_pixels / (2.0 * area) <= 0.1) and (roundness >= 0.7)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def remove_noise(
    cnp.ndarray[cnp.int32_t, ndim=2] cc,
    cnp.ndarray[cnp.int32_t, ndim=2] cc_stats,
    double average_height,
    double median_width,
):
    """
    Remove noise from detected connected components
    :param cc: connected components labels array
    :param cc_stats: connected components' statistics array
    :param average_height: average connected components' height
    :param median_width: median connected components' width
    :return: connected components labels array without noisy components
    """
    cdef cnp.int32_t[:, ::1] cc_view = cc
    cdef cnp.int32_t[:, ::1] stats_view = cc_stats
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] keep_label = np.ones(len(cc_stats), dtype=np.uint8)
    cdef cnp.uint8_t[::1] keep_view = keep_label
    cdef Py_ssize_t idx, row, col
    cdef long w_cc, h_cc, area, label
    cdef bint is_dash, is_dot
    cdef double elongation, density

    # Create lookup table of connected components labels to keep
    keep_view[0] = 0

    for idx in range(1, len(cc_stats)):
        w_cc = stats_view[idx, 2]
        h_cc = stats_view[idx, 3]
        area = stats_view[idx, 4]

        # Check dashes
        is_dash = (w_cc / <double> h_cc >= 2.0) and (0.5 * median_width <= w_cc <= 1.5 * median_width)
        is_dot = _is_dot_component(cc=cc, cc_stats=cc_stats, idx=idx)
        if is_dash or is_dot:
            continue

        # Metrics
        elongation = _min_long(h_cc, w_cc) / <double> _max_long(h_cc, w_cc if w_cc != 0 else 1)
        density = area / <double> (_max_long(w_cc, 1) * _max_long(h_cc, 1))

        # Mark for removal
        if h_cc < (average_height / 3.0) or density < 0.08 or elongation < 0.08:
            keep_view[idx] = 0

    for row in range(cc.shape[0]):
        for col in range(cc.shape[1]):
            label = cc_view[row, col]
            if label > 0 and not keep_view[label]:
                cc_view[row, col] = 0

    return cc


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def adaptive_rlsa(
    cnp.ndarray[cnp.int32_t, ndim=2] cc,
    cnp.ndarray[cnp.int32_t, ndim=2] cc_stats,
    cnp.ndarray[cnp.uint8_t, ndim=2] obstacle_mask,
    double a,
    double th,
    double c,
):
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
    cdef cnp.int32_t[:, ::1] cc_view = cc
    cdef cnp.int32_t[:, ::1] stats_view = cc_stats
    cdef cnp.uint8_t[:, ::1] obstacle_view = obstacle_mask
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] rlsa_img = np.zeros((cc.shape[0], cc.shape[1]), dtype=np.uint8)
    cdef cnp.uint8_t[:, ::1] rlsa_view = rlsa_img
    cdef Py_ssize_t h, w, row, col, x, ny, nx
    cdef long prev_cc_position, prev_cc_label, label, length
    cdef long y1_cc, h_cc, y1_prev, h_prev, cc_value
    cdef double height_ratio, h_overlap
    cdef bint crosses_obstacle, no_other_cc

    h = cc.shape[0]
    w = cc.shape[1]
    for row in range(h):
        for col in range(w):
            if cc_view[row, col] > 0:
                rlsa_view[row, col] = 1

    for row in range(h):
        prev_cc_position = -1
        prev_cc_label = 0

        for col in range(w):
            label = cc_view[row, col]
            if label == 0:
                continue

            if prev_cc_label == 0:
                prev_cc_position = col
                prev_cc_label = label
                continue

            # Pixels between two points of the same component are always filled.
            if label == prev_cc_label:
                for x in range(prev_cc_position, col):
                    rlsa_view[row, x] = 1
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
                if obstacle_view[row, x] > 0:
                    crosses_obstacle = True
                    break
            if crosses_obstacle:
                prev_cc_position = col
                prev_cc_label = label
                continue

            # Geometric constraints
            y1_cc = stats_view[label, 1]
            h_cc = stats_view[label, 3]
            y1_prev = stats_view[prev_cc_label, 1]
            h_prev = stats_view[prev_cc_label, 3]
            height_ratio = _max_long(h_cc, h_prev) / <double> _max_long(_min_long(h_cc, h_prev), 1)
            h_overlap = _min_long(y1_cc + h_cc, y1_prev + h_prev) - _max_long(y1_cc, y1_prev)

            # Check for other CC in the 3x3 neighborhood of sequence pixels.
            no_other_cc = True
            for x in range(prev_cc_position + 1, col):
                for ny in range(_max_long(0, row - 1), _min_long(h, row + 2)):
                    for nx in range(_max_long(0, x - 1), _min_long(w, x + 2)):
                        cc_value = cc_view[ny, nx]
                        if cc_value != 0 and cc_value != label and cc_value != prev_cc_label:
                            no_other_cc = False
                            break
                    if not no_other_cc:
                        break
                if not no_other_cc:
                    break

            if (
                length <= a * _min_long(h_cc, h_prev)
                and height_ratio <= th
                and h_overlap >= c * _min_long(h_cc, h_prev)
                and no_other_cc
            ):
                for x in range(prev_cc_position, col):
                    rlsa_view[row, x] = 1

            prev_cc_position = col
            prev_cc_label = label

    return rlsa_img


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _remove_punctuation_marks(
    cnp.ndarray[cnp.uint8_t, ndim=2] thresh,
    cnp.ndarray[cnp.int32_t, ndim=2] labels,
    cnp.ndarray[cnp.int32_t, ndim=2] stats,
):
    """
    Create images with punctuation marks removed as well as image of punctuation marks
    :param thresh: input threshold image
    :param labels: connected components labels
    :param stats: connected components stats
    """
    cdef cnp.uint8_t[:, ::1] thresh_view = thresh
    cdef cnp.int32_t[:, ::1] labels_view = labels
    cdef cnp.int32_t[:, ::1] stats_view = stats
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] cleaned = np.zeros((thresh.shape[0], thresh.shape[1]), dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] punctuation = np.zeros((thresh.shape[0], thresh.shape[1]), dtype=np.uint8)
    cdef cnp.uint8_t[:, ::1] cleaned_view = cleaned
    cdef cnp.uint8_t[:, ::1] punctuation_view = punctuation
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] non_zeros = np.zeros(stats.shape[0], dtype=np.uint16)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] is_punct_map = np.zeros(stats.shape[0], dtype=np.uint8)
    cdef cnp.uint16_t[::1] non_zero_view = non_zeros
    cdef cnp.uint8_t[::1] punct_view = is_punct_map
    cdef Py_ssize_t row, col, cc_idx
    cdef long area
    cdef cnp.uint8_t val

    # Compute number of non zeros elements by labels
    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            cc_idx = labels_view[row, col]
            if cc_idx > 0 and thresh_view[row, col] > 0:
                non_zero_view[cc_idx] += 1

    # Identify punctuation connected components
    for cc_idx in range(1, stats.shape[0]):
        area = stats_view[cc_idx, 4]

        # Assess if element is punctuation
        if non_zero_view[cc_idx] > 0 and area / <double> non_zero_view[cc_idx] <= 1.15:
            punct_view[cc_idx] = 1

    # Add to punctuation or cleaned image
    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            cc_idx = labels_view[row, col]
            if cc_idx > 0:
                val = thresh_view[row, col]
                if punct_view[cc_idx]:
                    punctuation_view[row, col] = val
                else:
                    cleaned_view[row, col] = val

    return cleaned, punctuation


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def detect_obstacles(cnp.ndarray[cnp.uint8_t, ndim=2] img, double min_width):
    """
    Identify obstacles (columns, line gaps) in image
    :param img: image array
    :param min_width: minimum width of obstacles
    :return: connected components labels array with obstacles identified
    """
    cdef cnp.uint8_t[:, ::1] img_view = img
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] mask_obstacles = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cdef cnp.uint8_t[:, ::1] mask_view = mask_obstacles
    cdef cnp.ndarray[cnp.int32_t, ndim=2] row_prefix = np.zeros((img.shape[0], img.shape[1] + 1), dtype=np.int32)
    cdef cnp.int32_t[:, ::1] prefix_view = row_prefix
    cdef Py_ssize_t h, w, col, row, idx, id_row
    cdef long prev_cc_position, length
    cdef int int_min_width

    int_min_width = _max_long(1, <long> ceil(min_width))
    h = img.shape[0]
    w = img.shape[1]

    for row in range(h):
        for col in range(w):
            prefix_view[row, col + 1] = prefix_view[row, col] + (img_view[row, col] > 0)

    for col in range(w - int_min_width + 1):
        row, prev_cc_position = 0, -1
        for row in range(h):
            # Not a CC
            if prefix_view[row, col + int_min_width] == prefix_view[row, col]:
                continue

            length = row - prev_cc_position - 1
            if length > h / 10:
                for id_row in range(prev_cc_position + 1, row):
                    for idx in range(int_min_width):
                        mask_view[id_row, col + idx] = 1

            # Update counters
            prev_cc_position = row

        # Check ending
        length = row + 1 - prev_cc_position - 1
        if length > h / 10:
            for id_row in range(prev_cc_position + 1, row + 1):
                for idx in range(int_min_width):
                    mask_view[id_row, col + idx] = 1

    return mask_obstacles
