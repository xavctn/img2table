cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport M_PI

cnp.import_array()


cdef inline double _max_double(double a, double b) noexcept:
    return a if a >= b else b


cdef inline double _min_double(double a, double b) noexcept:
    return a if a <= b else b


cdef inline long _max_long(long a, long b) noexcept:
    return a if a >= b else b


cdef inline long _min_long(long a, long b) noexcept:
    return a if a <= b else b


cdef inline long _abs_long(long value) noexcept:
    return value if value >= 0 else -value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def remove_dots(
    cnp.ndarray[cnp.int32_t, ndim=2] cc_labels,
    cnp.ndarray[cnp.int32_t, ndim=2] stats,
):
    """
    Remove dots from connected components
    :param cc_labels: connected components' label array
    :param stats: connected components' stats array
    :return: list of non-dot connected components' indexes
    """
    cdef Py_ssize_t idx, row, col, out_count, cc_width, row_offset
    cdef cnp.int32_t[:, ::1] cc_view = cc_labels
    cdef cnp.int32_t[:, ::1] stats_view = stats
    cdef cnp.ndarray[cnp.int32_t, ndim=2] out = np.empty((max(stats.shape[0] - 1, 0), 5), dtype=np.int32)
    cdef cnp.int32_t[:, ::1] out_view = out
    cdef long x, y, w, h, area, inner_pixels, prev_position
    cdef long max_dim
    cdef double roundness

    out_count = 0
    cc_width = cc_labels.shape[1]

    for idx in range(stats.shape[0]):
        if idx == 0:
            continue

        x = stats_view[idx, 0]
        y = stats_view[idx, 1]
        w = stats_view[idx, 2]
        h = stats_view[idx, 3]
        area = stats_view[idx, 4]

        # Check number of inner pixels
        inner_pixels = 0
        for row in range(y, y + h):
            prev_position = -1
            row_offset = row * cc_width
            for col in range(x, x + w):
                if cc_view[row, col] == idx:
                    if prev_position >= 0:
                        inner_pixels += col - prev_position - 1
                    prev_position = col

        for col in range(x, x + w):
            prev_position = -1
            for row in range(y, y + h):
                if cc_view[row, col] == idx:
                    if prev_position >= 0:
                        inner_pixels += row - prev_position - 1
                    prev_position = row

        # Compute roundness
        max_dim = h if h >= w else w
        roundness = 4.0 * area / (M_PI * max_dim * max_dim)

        if not (inner_pixels / (2.0 * area) <= 0.1 and roundness >= 0.7):
            out_view[out_count, 0] = <cnp.int32_t> x
            out_view[out_count, 1] = <cnp.int32_t> y
            out_view[out_count, 2] = <cnp.int32_t> w
            out_view[out_count, 3] = <cnp.int32_t> h
            out_view[out_count, 4] = <cnp.int32_t> area
            out_count += 1

    return out[:out_count]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_interval_union_length(
    cnp.ndarray[cnp.float64_t, ndim=1] starts,
    cnp.ndarray[cnp.float64_t, ndim=1] ends,
    int interval_count,
):
    """
    Compute union length of projected intervals
    :param starts: interval starts
    :param ends: interval ends
    :param interval_count: number of valid intervals
    :return: union length
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sorted_starts
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sorted_ends
    cdef cnp.float64_t[::1] sorted_starts_view
    cdef cnp.float64_t[::1] sorted_ends_view
    cdef Py_ssize_t idx, prev_idx
    cdef double start, end, current_start, current_end, union_length

    if interval_count == 0:
        return 0.0

    sorted_starts = np.empty(interval_count, dtype=np.float64)
    sorted_ends = np.empty(interval_count, dtype=np.float64)
    sorted_starts_view = sorted_starts
    sorted_ends_view = sorted_ends

    for idx in range(interval_count):
        sorted_starts_view[idx] = starts[idx]
        sorted_ends_view[idx] = ends[idx]

    for idx in range(1, interval_count):
        start = sorted_starts_view[idx]
        end = sorted_ends_view[idx]
        prev_idx = idx - 1

        while prev_idx >= 0 and sorted_starts_view[prev_idx] > start:
            sorted_starts_view[prev_idx + 1] = sorted_starts_view[prev_idx]
            sorted_ends_view[prev_idx + 1] = sorted_ends_view[prev_idx]
            prev_idx -= 1

        sorted_starts_view[prev_idx + 1] = start
        sorted_ends_view[prev_idx + 1] = end

    current_start = sorted_starts_view[0]
    current_end = sorted_ends_view[0]
    union_length = 0.0

    for idx in range(1, interval_count):
        start = sorted_starts_view[idx]
        end = sorted_ends_view[idx]

        if start <= current_end:
            current_end = _max_double(current_end, end)
        else:
            union_length += current_end - current_start
            current_start = start
            current_end = end

    return union_length + current_end - current_start


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def remove_dotted_lines(cnp.ndarray[cnp.float64_t, ndim=2] complete_stats):
    """
    Remove dotted lines in image by identifying aligned connected components
    :param complete_stats: connected components' array
    :return: filtered connected components' array
    """
    cdef cnp.ndarray sorted_stats
    cdef cnp.ndarray areas_array
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_starts
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_ends
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y_starts
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y_ends
    cdef cnp.ndarray[cnp.float64_t, ndim=2] line_areas = np.empty((max(2 * complete_stats.shape[0], 1), 4), dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] kept = np.empty((complete_stats.shape[0], 5), dtype=np.int32)
    cdef cnp.ndarray[cnp.intp_t, ndim=1] order
    cdef cnp.float64_t[:, ::1] stats_view
    cdef cnp.float64_t[:, ::1] line_areas_view = line_areas
    cdef cnp.int32_t[:, ::1] kept_view = kept
    cdef Py_ssize_t idx, j, area_count, line_count, kept_count
    cdef double x, y, w, h, area, x_middle, y_middle
    cdef double x1_area, y1_area, x2_area, y2_area, prev_y_middle, prev_x_middle
    cdef double width_area, height_area, intersection_area, x1_area_i, y1_area_i, x2_area_i, y2_area_i
    cdef double x_overlap, y_overlap, denom

    ### Identify horizontal lines
    order = np.argsort(complete_stats[:, 6])
    sorted_stats = np.ascontiguousarray(complete_stats[order], dtype=np.float64)
    stats_view = sorted_stats

    x_starts = np.empty(sorted_stats.shape[0], dtype=np.float64)
    x_ends = np.empty(sorted_stats.shape[0], dtype=np.float64)
    x1_area, y1_area, x2_area, y2_area, prev_y_middle, area_count = 0.0, 0.0, 0.0, 0.0, -10.0, 0
    line_count = 0
    for idx in range(sorted_stats.shape[0]):
        x = stats_view[idx, 0]
        y = stats_view[idx, 1]
        w = stats_view[idx, 2]
        h = stats_view[idx, 3]
        x_middle = stats_view[idx, 5]
        y_middle = stats_view[idx, 6]

        if w / h < 2.0:
            continue

        if y_middle - prev_y_middle <= 2.0:
            # Add to previous area
            x1_area = _min_double(x, x1_area)
            y1_area = _min_double(y, y1_area)
            x2_area = _max_double(x + w, x2_area)
            y2_area = _max_double(y + h, y2_area)
            x_starts[area_count] = x
            x_ends[area_count] = x + w
            area_count += 1
            prev_y_middle = y_middle
        else:
            # Check if previously defined area is relevant
            width_area = compute_interval_union_length(
                starts=x_starts,
                ends=x_ends,
                interval_count=<int> area_count,
            )
            if area_count >= 5 and width_area / ((x2_area - x1_area) or 1.0) >= 0.66:
                line_areas_view[line_count, 0] = x1_area
                line_areas_view[line_count, 1] = y1_area
                line_areas_view[line_count, 2] = x2_area
                line_areas_view[line_count, 3] = y2_area
                line_count += 1
            # Create new area
            x1_area, y1_area, x2_area, y2_area = x, y, x + w, y + h
            x_starts[0] = x
            x_ends[0] = x + w
            prev_y_middle, area_count = y_middle, 1

    # Check last area
    width_area = compute_interval_union_length(starts=x_starts, ends=x_ends, interval_count=<int> area_count)
    if area_count >= 5 and width_area / ((x2_area - x1_area) or 1.0) >= 0.66:
        line_areas_view[line_count, 0] = x1_area
        line_areas_view[line_count, 1] = y1_area
        line_areas_view[line_count, 2] = x2_area
        line_areas_view[line_count, 3] = y2_area
        line_count += 1

    ### Identify vertical lines
    order = np.argsort(sorted_stats[:, 5])
    sorted_stats = np.ascontiguousarray(sorted_stats[order], dtype=np.float64)
    stats_view = sorted_stats

    y_starts = np.empty(sorted_stats.shape[0], dtype=np.float64)
    y_ends = np.empty(sorted_stats.shape[0], dtype=np.float64)
    x1_area, y1_area, x2_area, y2_area, prev_x_middle, area_count = 0.0, 0.0, 0.0, 0.0, -10.0, 0
    for idx in range(sorted_stats.shape[0]):
        x = stats_view[idx, 0]
        y = stats_view[idx, 1]
        w = stats_view[idx, 2]
        h = stats_view[idx, 3]
        x_middle = stats_view[idx, 5]
        y_middle = stats_view[idx, 6]

        if h / w < 2.0:
            continue

        if x_middle - prev_x_middle <= 2.0:
            # Add to previous area
            x1_area = _min_double(x, x1_area)
            y1_area = _min_double(y, y1_area)
            x2_area = _max_double(x + w, x2_area)
            y2_area = _max_double(y + h, y2_area)
            y_starts[area_count] = y
            y_ends[area_count] = y + h
            area_count += 1
            prev_x_middle = x_middle
        else:
            # Check if previously defined area is relevant
            height_area = compute_interval_union_length(
                starts=y_starts,
                ends=y_ends,
                interval_count=<int> area_count,
            )
            if area_count >= 5 and height_area / ((y2_area - y1_area) or 1.0) >= 0.66:
                line_areas_view[line_count, 0] = x1_area
                line_areas_view[line_count, 1] = y1_area
                line_areas_view[line_count, 2] = x2_area
                line_areas_view[line_count, 3] = y2_area
                line_count += 1
            # Create new area
            x1_area, y1_area, x2_area, y2_area = x, y, x + w, y + h
            y_starts[0] = y
            y_ends[0] = y + h
            prev_x_middle, area_count = x_middle, 1

    # Check last area
    height_area = compute_interval_union_length(starts=y_starts, ends=y_ends, interval_count=<int> area_count)
    if area_count >= 5 and height_area / ((y2_area - y1_area) or 1.0) >= 0.66:
        line_areas_view[line_count, 0] = x1_area
        line_areas_view[line_count, 1] = y1_area
        line_areas_view[line_count, 2] = x2_area
        line_areas_view[line_count, 3] = y2_area
        line_count += 1

    if line_count == 0:
        return sorted_stats[:, :5].astype(np.int32)

    # Create array of line areas
    areas_array = line_areas[:line_count]
    line_areas_view = areas_array

    # Check if connected components is located in areas
    kept_count = 0
    for idx in range(sorted_stats.shape[0]):
        x = stats_view[idx, 0]
        y = stats_view[idx, 1]
        w = stats_view[idx, 2]
        h = stats_view[idx, 3]
        area = stats_view[idx, 4]

        intersection_area = 0.0
        for j in range(line_count):
            x1_area_i = line_areas_view[j, 0]
            y1_area_i = line_areas_view[j, 1]
            x2_area_i = line_areas_view[j, 2]
            y2_area_i = line_areas_view[j, 3]

            # Compute overlaps
            x_overlap = _max_double(0.0, _min_double(x2_area_i, x + w) - _max_double(x1_area_i, x))
            y_overlap = _max_double(0.0, _min_double(y2_area_i, y + h) - _max_double(y1_area_i, y))
            intersection_area += x_overlap * y_overlap

        denom = w * h
        if intersection_area / denom < 0.25:
            kept_view[kept_count, 0] = <cnp.int32_t> x
            kept_view[kept_count, 1] = <cnp.int32_t> y
            kept_view[kept_count, 2] = <cnp.int32_t> w
            kept_view[kept_count, 3] = <cnp.int32_t> h
            kept_view[kept_count, 4] = <cnp.int32_t> area
            kept_count += 1

    return kept[:kept_count]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def filter_cc(cnp.ndarray[cnp.int32_t, ndim=2] stats):
    """
    Filter relevant connected components
    :param stats: connected components' array
    :return: tuple with relevant connected components' array and discarded connected components' array
    """
    cdef Py_ssize_t idx, kept_count, discarded_count, final_kept_count
    cdef cnp.int32_t[:, ::1] stats_view = stats
    cdef cnp.ndarray[cnp.int32_t, ndim=2] kept_first = np.empty((stats.shape[0], 5), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] discarded = np.empty((stats.shape[0], 5), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] kept_final = np.empty((stats.shape[0], 5), dtype=np.int32)
    cdef cnp.int32_t[:, ::1] kept_first_view = kept_first
    cdef cnp.int32_t[:, ::1] discarded_view = discarded
    cdef cnp.int32_t[:, ::1] kept_final_view = kept_final
    cdef long x, y, w, h, area
    cdef double ar, fill, median_width, median_height, upper_bound, lower_bound
    cdef bint bounded_area, is_dash

    kept_count = 0
    discarded_count = 0

    for idx in range(stats.shape[0]):
        x = stats_view[idx, 0]
        y = stats_view[idx, 1]
        w = stats_view[idx, 2]
        h = stats_view[idx, 3]
        area = stats_view[idx, 4]

        # Compute aspect ratio and fill ratio
        ar = _max_long(w, h) / <double> _min_long(w, h)
        fill = area / <double> (w * h)

        if ar <= 5.0 and fill > 0.08:
            kept_first_view[kept_count, 0] = <cnp.int32_t> x
            kept_first_view[kept_count, 1] = <cnp.int32_t> y
            kept_first_view[kept_count, 2] = <cnp.int32_t> w
            kept_first_view[kept_count, 3] = <cnp.int32_t> h
            kept_first_view[kept_count, 4] = <cnp.int32_t> area
            kept_count += 1
        else:
            discarded_view[discarded_count, 0] = <cnp.int32_t> x
            discarded_view[discarded_count, 1] = <cnp.int32_t> y
            discarded_view[discarded_count, 2] = <cnp.int32_t> w
            discarded_view[discarded_count, 3] = <cnp.int32_t> h
            discarded_view[discarded_count, 4] = <cnp.int32_t> area
            discarded_count += 1

    if kept_count == 0:
        # Map to arrays
        return kept_first[:0], discarded[:discarded_count]

    # Map kept_cc to array and compute metrics
    median_width = np.median(kept_first[:kept_count, 2])
    median_height = np.median(kept_first[:kept_count, 3])

    # Compute bbox area bounds
    upper_bound = 5.0 * median_width * median_height
    lower_bound = 0.2 * median_width * median_height

    final_kept_count = 0
    for idx in range(kept_count):
        x = kept_first_view[idx, 0]
        y = kept_first_view[idx, 1]
        w = kept_first_view[idx, 2]
        h = kept_first_view[idx, 3]
        area = kept_first_view[idx, 4]

        # Check area
        bounded_area = lower_bound <= w * h <= upper_bound
        # Check dashes
        is_dash = (w / <double> h >= 2.0) and (0.5 * median_width <= w <= 1.5 * median_width)

        if bounded_area or is_dash:
            kept_final_view[final_kept_count, 0] = <cnp.int32_t> x
            kept_final_view[final_kept_count, 1] = <cnp.int32_t> y
            kept_final_view[final_kept_count, 2] = <cnp.int32_t> w
            kept_final_view[final_kept_count, 3] = <cnp.int32_t> h
            kept_final_view[final_kept_count, 4] = <cnp.int32_t> area
            final_kept_count += 1
        else:
            discarded_view[discarded_count, 0] = <cnp.int32_t> x
            discarded_view[discarded_count, 1] = <cnp.int32_t> y
            discarded_view[discarded_count, 2] = <cnp.int32_t> w
            discarded_view[discarded_count, 3] = <cnp.int32_t> h
            discarded_view[discarded_count, 4] = <cnp.int32_t> area
            discarded_count += 1

    # Map to arrays
    return kept_final[:final_kept_count], discarded[:discarded_count]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def create_character_thresh(
    cnp.ndarray[cnp.uint8_t, ndim=2] thresh,
    cnp.ndarray[cnp.int32_t, ndim=2] stats,
    cnp.ndarray[cnp.int32_t, ndim=2] discarded_stats,
    double char_length,
):
    """
    Create thresholded image containing uniquely characters
    :param thresh: thresholded image
    :param stats: relevant connected components' array
    :param discarded_stats: discarded connected components' array
    :param char_length: average character length
    :return: thresholded image containing uniquely characters and array of image characters
    """
    cdef Py_ssize_t idx, idx_discarded, row, col, out_count, max_rows
    cdef long x, y, w, h, area, cc_x, cc_y, cc_w, cc_h, cc_area
    cdef double y_overlap, distance
    cdef cnp.uint8_t[:, ::1] thresh_view = thresh
    cdef cnp.int32_t[:, ::1] stats_view = stats
    cdef cnp.int32_t[:, ::1] discarded_view = discarded_stats
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] character_thresh = np.zeros((thresh.shape[0], thresh.shape[1]), dtype=np.uint8)
    cdef cnp.uint8_t[:, ::1] character_view = character_thresh
    cdef cnp.ndarray[cnp.int32_t, ndim=2] relevant_chars
    cdef cnp.int32_t[:, ::1] relevant_view

    # Create blank character thresh
    max_rows = stats.shape[0] * (discarded_stats.shape[0] if discarded_stats.shape[0] > 1 else 1)
    relevant_chars = np.empty((max_rows, 5), dtype=np.int32)
    relevant_view = relevant_chars
    out_count = 0

    # Identify CC from discarded connected components that can be characters
    for idx in range(stats.shape[0]):
        x = stats_view[idx, 0]
        y = stats_view[idx, 1]
        w = stats_view[idx, 2]
        h = stats_view[idx, 3]
        area = stats_view[idx, 4]

        # Add character to thresholded image
        relevant_view[out_count, 0] = <cnp.int32_t> x
        relevant_view[out_count, 1] = <cnp.int32_t> y
        relevant_view[out_count, 2] = <cnp.int32_t> w
        relevant_view[out_count, 3] = <cnp.int32_t> h
        relevant_view[out_count, 4] = <cnp.int32_t> area
        out_count += 1
        for row in range(y, y + h):
            for col in range(x, x + w):
                character_view[row, col] = thresh_view[row, col]

        for idx_discarded in range(1, discarded_stats.shape[0]):
            cc_x = discarded_view[idx_discarded, 0]
            cc_y = discarded_view[idx_discarded, 1]
            cc_w = discarded_view[idx_discarded, 2]
            cc_h = discarded_view[idx_discarded, 3]
            cc_area = discarded_view[idx_discarded, 4]

            # Compute y overlap
            y_overlap = _min_long(cc_y + cc_h, y + h) - _max_long(cc_y, y)

            if y_overlap < 0.5 * _min_long(cc_h, h):
                continue
            if _max_long(cc_h, cc_w) > 3 * _max_long(h, w):
                continue

            # Compute horizontal distance
            distance = _abs_long(cc_x - x)
            distance = _min_double(distance, _abs_long(cc_x - x - w))
            distance = _min_double(distance, _abs_long(cc_x + cc_w - x))
            distance = _min_double(distance, _abs_long(cc_x + cc_w - x - w))

            if y_overlap > 0.0 and distance <= char_length:
                # Add new character to thresholded image
                relevant_view[out_count, 0] = <cnp.int32_t> cc_x
                relevant_view[out_count, 1] = <cnp.int32_t> cc_y
                relevant_view[out_count, 2] = <cnp.int32_t> cc_w
                relevant_view[out_count, 3] = <cnp.int32_t> cc_h
                relevant_view[out_count, 4] = <cnp.int32_t> cc_area
                out_count += 1
                for row in range(cc_y, cc_y + cc_h):
                    for col in range(cc_x, cc_x + cc_w):
                        character_view[row, col] = thresh_view[row, col]

    return character_thresh, relevant_chars[:out_count]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def recompute_contours(
    cnp.ndarray[cnp.int32_t, ndim=2] stats,
    cnp.ndarray[cnp.int32_t, ndim=2] chars_array,
):
    """
    Recompute contours from CC analysis with original characters
    :param stats: contours from CC analysis
    :param chars_array: characters array
    :return: array of contours with dimensions recomputed
    """
    cdef Py_ssize_t idx, id_c, out_count
    cdef cnp.int32_t[:, ::1] stats_view = stats
    cdef cnp.int32_t[:, ::1] chars_view = chars_array
    cdef cnp.ndarray[cnp.int64_t, ndim=2] out = np.empty((max(stats.shape[0] - 1, 0), 4), dtype=np.int64)
    cdef cnp.int64_t[:, ::1] out_view = out
    cdef long x, y, w, h, xc, yc, wc, hc
    cdef long x1, y1, x2, y2, nb_chars, x_overlap, y_overlap

    out_count = 0
    for idx in range(stats.shape[0]):
        if idx == 0:
            continue
        x = stats_view[idx, 0]
        y = stats_view[idx, 1]
        w = stats_view[idx, 2]
        h = stats_view[idx, 3]

        # Identify contour coordinates by matching included characters
        x1, y1, x2, y2, nb_chars = 10**6, 10**6, 0, 0, 0
        for id_c in range(chars_array.shape[0]):
            xc = chars_view[id_c, 0]
            yc = chars_view[id_c, 1]
            wc = chars_view[id_c, 2]
            hc = chars_view[id_c, 3]

            # Compute overlaps
            x_overlap = _max_long(0, _min_long(x + w, xc + wc) - _max_long(x, xc))
            y_overlap = _max_long(0, _min_long(y + h, yc + hc) - _max_long(y, yc))

            if x_overlap * y_overlap >= 0.5 * hc * wc:
                # Update stats
                x1 = _min_long(x1, xc)
                y1 = _min_long(y1, yc)
                x2 = _max_long(x2, xc + wc)
                y2 = _max_long(y2, yc + hc)
                nb_chars += 1

        if nb_chars > 0:
            out_view[out_count, 0] = x1
            out_view[out_count, 1] = y1
            out_view[out_count, 2] = x2 - x1
            out_view[out_count, 3] = y2 - y1
            out_count += 1

    return out[:out_count]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_row_separations(cnp.ndarray[cnp.int64_t, ndim=2] stats, double char_length):
    """
    Compute row separation between contours
    :param stats: array of contours
    :param char_length: average character length
    :return: list of row separations
    """
    cdef Py_ssize_t i, j, out_count
    cdef cnp.int64_t[:, ::1] stats_view = stats
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(stats.shape[0], dtype=np.float64)
    cdef cnp.float64_t[::1] out_view = out
    cdef long xi, yi, wi, hi, xj, yj, wj, hj
    cdef double row_separation, h_overlap, v_pos_i, v_pos_j, half_char

    out_count = 0
    half_char = char_length // 2.0

    for i in range(stats.shape[0]):
        # Get statistics
        xi = stats_view[i, 0]
        yi = stats_view[i, 1]
        wi = stats_view[i, 2]
        hi = stats_view[i, 3]
        row_separation = 10**6

        for j in range(stats.shape[0]):
            if i == j:
                continue

            # Get statistics
            xj = stats_view[j, 0]
            yj = stats_view[j, 1]
            wj = stats_view[j, 2]
            hj = stats_view[j, 3]

            # Compute horizontal overlap and vertical positions
            h_overlap = _min_long(xi + wi, xj + wj) - _max_long(xi, xj)
            v_pos_i = (2.0 * yi + hi) / 2.0
            v_pos_j = (2.0 * yj + hj) / 2.0
            if h_overlap <= half_char or v_pos_j <= v_pos_i:
                continue

            row_separation = _min_double(row_separation, v_pos_j - v_pos_i)

        if row_separation < 10**6:
            out_view[out_count] = row_separation
            out_count += 1

    return out[:out_count]
