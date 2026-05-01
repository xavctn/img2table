cimport cython
cimport numpy as cnp
import numpy as np

cnp.import_array()


cdef inline long _max_long(long a, long b) noexcept:
    return a if a >= b else b


cdef inline long _min_long(long a, long b) noexcept:
    return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def identify_cells(
    cnp.ndarray[cnp.int64_t, ndim=2] h_lines_arr,
    cnp.ndarray[cnp.int64_t, ndim=2] v_lines_arr,
):
    """
    Identify cells from lines
    :param h_lines_arr: array containing horizontal lines
    :param v_lines_arr: array containing vertical lines
    :return: array of cells coordinates
    """
    cdef cnp.int64_t[:, ::1] h_view = h_lines_arr
    cdef cnp.int64_t[:, ::1] v_view = v_lines_arr
    cdef Py_ssize_t i, j, idx, potential_count, upper_count, lower_count, delim_count, cell_count
    cdef long x1i, y1i, x2i, x1j, y1j, x2j, y2j
    cdef long x1, x2, y1, y2, x1v, y1v, y2v, overlap
    cdef double margin, tolerance
    cdef long prev_x1, prev_x2, prev_y1, prev_y2
    cdef bint l_corresponds, r_corresponds, l_contained, r_contained
    cdef cnp.ndarray[cnp.int64_t, ndim=2] potential_cells
    cdef cnp.ndarray[cnp.int64_t, ndim=2] dedup_upper
    cdef cnp.ndarray[cnp.int64_t, ndim=2] dedup_lower
    cdef cnp.ndarray[cnp.int64_t, ndim=2] cells
    cdef cnp.ndarray[cnp.int64_t, ndim=1] delimiters
    cdef cnp.ndarray[cnp.intp_t, ndim=1] order
    cdef cnp.int64_t[:, ::1] potential_view
    cdef cnp.int64_t[:, ::1] upper_view
    cdef cnp.int64_t[:, ::1] lower_view
    cdef cnp.int64_t[:, ::1] cells_view
    cdef cnp.int64_t[::1] delimiters_view

    # Get potential cells from horizontal lines
    potential_cells = np.empty((h_lines_arr.shape[0] * h_lines_arr.shape[0], 4), dtype=np.int64)
    potential_view = potential_cells
    potential_count = 0
    for i in range(h_lines_arr.shape[0]):
        x1i = h_view[i, 0]
        y1i = h_view[i, 1]
        x2i = h_view[i, 2]
        for j in range(h_lines_arr.shape[0]):
            x1j = h_view[j, 0]
            y1j = h_view[j, 1]
            x2j = h_view[j, 2]
            y2j = h_view[j, 3]

            if y1i >= y1j:
                continue

            # Check correspondence between lines
            l_corresponds = -0.02 <= (x1i - x1j) / <double> ((x2i - x1i) or 1) <= 0.02
            r_corresponds = -0.02 <= (x2i - x2j) / <double> ((x2i - x1i) or 1) <= 0.02
            l_contained = (x1i <= x1j <= x2i) or (x1j <= x1i <= x2j)
            r_contained = (x1i <= x2j <= x2i) or (x1j <= x2i <= x2j)

            if (l_corresponds or l_contained) and (r_corresponds or r_contained):
                potential_view[potential_count, 0] = _max_long(x1i, x1j)
                potential_view[potential_count, 1] = _min_long(x2i, x2j)
                potential_view[potential_count, 2] = y1i
                potential_view[potential_count, 3] = y2j
                potential_count += 1

    if potential_count == 0:
        return np.empty((0, 4), dtype=np.int64)

    # Deduplicate on upper bound
    order = np.lexsort((
        potential_cells[:potential_count, 3],
        potential_cells[:potential_count, 2],
        potential_cells[:potential_count, 1],
        potential_cells[:potential_count, 0],
    ))
    potential_cells = np.ascontiguousarray(potential_cells[:potential_count][order], dtype=np.int64)
    potential_view = potential_cells

    dedup_upper = np.empty((potential_count, 4), dtype=np.int64)
    upper_view = dedup_upper
    upper_count = 0
    prev_x1, prev_x2, prev_y1 = 0, 0, 0
    for idx in range(potential_count):
        x1 = potential_view[idx, 0]
        x2 = potential_view[idx, 1]
        y1 = potential_view[idx, 2]
        y2 = potential_view[idx, 3]

        if not (x1 == prev_x1 and x2 == prev_x2 and y1 == prev_y1):
            upper_view[upper_count, 0] = x1
            upper_view[upper_count, 1] = x2
            upper_view[upper_count, 2] = y2
            upper_view[upper_count, 3] = -y1
            upper_count += 1
        prev_x1, prev_x2, prev_y1 = x1, x2, y1

    # Deduplicate on lower bound
    order = np.lexsort((
        dedup_upper[:upper_count, 3],
        dedup_upper[:upper_count, 2],
        dedup_upper[:upper_count, 1],
        dedup_upper[:upper_count, 0],
    ))
    dedup_upper = np.ascontiguousarray(dedup_upper[:upper_count][order], dtype=np.int64)
    upper_view = dedup_upper

    dedup_lower = np.empty((upper_count, 4), dtype=np.int64)
    lower_view = dedup_lower
    lower_count = 0
    prev_x1, prev_x2, prev_y2 = 0, 0, 0
    for idx in range(upper_count):
        x1 = upper_view[idx, 0]
        x2 = upper_view[idx, 1]
        y2 = upper_view[idx, 2]
        y1 = -upper_view[idx, 3]

        if not (x1 == prev_x1 and x2 == prev_x2 and y2 == prev_y2):
            lower_view[lower_count, 0] = x1
            lower_view[lower_count, 1] = x2
            lower_view[lower_count, 2] = y1
            lower_view[lower_count, 3] = y2
            lower_count += 1
        prev_x1, prev_x2, prev_y2 = x1, x2, y2

    # Create array of potential cells
    cells = np.empty((lower_count * _max_long(v_lines_arr.shape[0] - 1, 0), 4), dtype=np.int64)
    cells_view = cells
    delimiters = np.empty(v_lines_arr.shape[0], dtype=np.int64)
    delimiters_view = delimiters
    cell_count = 0

    for i in range(lower_count):
        x1 = lower_view[i, 0]
        x2 = lower_view[i, 1]
        y1 = lower_view[i, 2]
        y2 = lower_view[i, 3]

        # Compute horizontal margin
        margin = _max_long(5, <long> ((x2 - x1) * 0.025))

        delim_count = 0
        for j in range(v_lines_arr.shape[0]):
            x1v = v_view[j, 0]
            y1v = v_view[j, 1]
            y2v = v_view[j, 3]

            if x1 - margin <= x1v <= x2 + margin:
                # Check vertical overlapping and tolerance
                overlap = _min_long(y2, y2v) - _max_long(y1, y1v)
                tolerance = _max_long(5, _min_long(10, <long> (0.1 * (y2 - y1))))

                if y2 - y1 - overlap <= tolerance:
                    delimiters_view[delim_count] = x1v
                    delim_count += 1

        # Create new cells from delimiters
        if delim_count >= 2:
            delimiters[:delim_count] = np.sort(delimiters[:delim_count])
            for j in range(delim_count - 1):
                cells_view[cell_count, 0] = delimiters_view[j]
                cells_view[cell_count, 1] = y1
                cells_view[cell_count, 2] = delimiters_view[j + 1]
                cells_view[cell_count, 3] = y2
                cell_count += 1

    return cells[:cell_count]
