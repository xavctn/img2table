
import numpy as np
from numba import njit, prange

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line


@njit("int64[:,:](int64[:,:],int64[:,:])", cache=True, fastmath=True)
def identify_cells(h_lines_arr: np.ndarray, v_lines_arr: np.ndarray) -> np.ndarray:
    """
    Identify cells from lines
    :param h_lines_arr: array containing horizontal lines
    :param v_lines_arr: array containing vertical lines
    :return: array of cells coordinates
    """
    # Get potential cells from horizontal lines
    potential_cells = []
    for i in prange(h_lines_arr.shape[0]):
        x1i, y1i, x2i, y2i = h_lines_arr[i][:]
        for j in prange(h_lines_arr.shape[0]):
            x1j, y1j, x2j, y2j = h_lines_arr[j][:]

            if y1i >= y1j:
                continue

            # Check correspondence between lines
            l_corresponds = -0.02 <= (x1i - x1j) / ((x2i - x1i) or 1) <= 0.02
            r_corresponds = -0.02 <= (x2i - x2j) / ((x2i - x1i) or 1) <= 0.02
            l_contained = (x1i <= x1j <= x2i) or (x1j <= x1i <= x2j)
            r_contained = (x1i <= x2j <= x2i) or (x1j <= x2i <= x2j)

            if (l_corresponds or l_contained) and (r_corresponds or r_contained):
                potential_cells.append([max(x1i, x1j), min(x2i, x2j), y1i, y2j])

    if len(potential_cells) == 0:
        return np.empty((0, 4), dtype=np.int64)

    # Deduplicate on upper bound
    potential_cells = sorted(potential_cells)
    dedup_upper = []
    prev_x1, prev_x2, prev_y1 = 0, 0, 0
    for idx in range(len(potential_cells)):
        x1, x2, y1, y2 = potential_cells[idx]

        if not (x1 == prev_x1 and x2 == prev_x2 and y1 == prev_y1):
            dedup_upper.append([x1, x2, y2, -y1])
        prev_x1, prev_x2, prev_y1 = x1, x2, y1

    # Deduplicate on lower bound
    dedup_upper = sorted(dedup_upper)
    dedup_lower = []
    prev_x1, prev_x2, prev_y2 = 0, 0, 0
    for idx in range(len(dedup_upper)):
        x1, x2, y2, _y1 = dedup_upper[idx]
        y1 = -_y1

        if not (x1 == prev_x1 and x2 == prev_x2 and y2 == prev_y2):
            dedup_lower.append([x1, x2, y1, y2])
        prev_x1, prev_x2, prev_y2 = x1, x2, y2

    # Create array of potential cells
    cells_array = np.array(dedup_lower)
    cells = []

    for i in prange(cells_array.shape[0]):
        x1, x2, y1, y2 = cells_array[i][:]

        # Compute horizontal margin
        margin = max(5, (x2 - x1) * 0.025)

        delimiters = []
        for j in range(v_lines_arr.shape[0]):
            x1v, y1v, x2v, y2v = v_lines_arr[j][:]

            if x1 - margin <= x1v <= x2 + margin:
                # Check vertical overlapping and tolerance
                overlap = min(y2, y2v) - max(y1, y1v)
                tolerance = max(5, min(10, 0.1 * (y2 - y1)))

                if y2 - y1 - overlap <= tolerance:
                    delimiters.append(x1v)

        # Create new cells from delimiters
        if len(delimiters) >= 2:
            delimiters = sorted(delimiters)
            for j in range(len(delimiters) - 1):
                cells.append([delimiters[j], y1, delimiters[j + 1], y2])

    return np.array(cells).astype(np.int64) if cells else np.empty((0, 4), dtype=np.int64)


def get_cells_dataframe(horizontal_lines: list[Line], vertical_lines: list[Line]) -> list[Cell]:
    """
    Create dataframe of all possible cells from horizontal and vertical rows
    :param horizontal_lines: list of horizontal rows
    :param vertical_lines: list of vertical rows
    :return: list of detected cells
    """
    # Check for empty rows
    if len(horizontal_lines) * len(vertical_lines) == 0:
        return []

    # Create arrays from horizontal and vertical rows
    h_lines_array = np.array([[line.x1, line.y1, line.x2, line.y2] for line in horizontal_lines], dtype=np.int64)
    v_lines_array = np.array([[line.x1, line.y1, line.x2, line.y2] for line in vertical_lines], dtype=np.int64)

    # Compute cells
    cells_array = identify_cells(h_lines_arr=h_lines_array,
                                 v_lines_arr=v_lines_array)

    return [Cell(x1=c[0], y1=c[1], x2=c[2], y2=c[3]) for c in cells_array]
