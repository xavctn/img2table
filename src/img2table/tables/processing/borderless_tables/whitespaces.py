
from typing import Union

import numpy as np
from numba import njit, prange

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import ImageSegment, ColumnGroup, Whitespace


@njit("List(List(List(int64)))(float64[:,:],float64,float64,float64,boolean)", cache=True, fastmath=True)
def compute_whitespaces(elements_array: np.ndarray, min_width: float, min_height: float, total_height: float,
                        continuous: bool = True) -> list[list[list[int]]]:
    """
    Compute whitespaces in segment
    :param elements_array: array of elements
    :param min_width: minimum width of the detected whitespaces
    :param min_height: minimum height of the detected whitespaces
    :param total_height: minimum total height of the continous/non-continous whitespaces
    :param continuous: boolean indicating if only continuous whitespaces are retrieved
    :return: list of groups of cells forming whitespaces
    """
    # Get x values in elements
    x_vals = set()
    for idx in prange(elements_array.shape[0]):
        x1, y1, x2, y2, y_middle = elements_array[idx][:]
        x_vals.add(x1)
        x_vals.add(x2)

    # Create array of x values
    x_array = np.transpose(np.array([list(x_vals)]))
    x_array = x_array[x_array[:, 0].argsort()]

    # Check ranges
    final_whitespaces = []
    for idx in prange(x_array.shape[0] - 1):
        x_min, x_max = x_array[idx][0], x_array[idx + 1][0]

        # Check array elements
        if x_max - x_min < min_width:
            continue

        # Identify whitespaces positions
        list_ws, prev_y = [], 10 ** 6
        for idx_el in range(elements_array.shape[0]):
            x1, y1, x2, y2, y_middle = elements_array[idx_el][:]

            # Check if it overlaps segment
            overlap = min(x_max, x2) - max(x_min, x1)
            if overlap > 0:
                # If whitespace is tall enough, add it
                if y1 - prev_y >= min_height:
                    list_ws.append([x_min, prev_y, x_max, y1])
                prev_y = y2

        # Create whitespaces
        if continuous:
            y_min, y_max = -1000, -1000
            for id_ws in range(len(list_ws)):
                x1_ws, y1_ws, x2_ws, y2_ws = list_ws[id_ws]

                # Check with previous ws
                if y1_ws == y_max:
                    y_min, y_max = min(y1_ws, y_min), max(y2_ws, y_max)
                else:
                    # Check current ws
                    if y_max - y_min >= total_height:
                        final_whitespaces.append([[int(x_min), int(y_min), int(x_max), int(y_max)]])
                    y_min, y_max = y1_ws, y2_ws

            # Check last whitespace
            if y_max - y_min >= total_height:
                final_whitespaces.append([[int(x_min), int(y_min), int(x_max), int(y_max)]])
        else:
            nb_ws, tot_height_ws, min_height_ws, max_height_ws = 0, 0, 10 ** 6, 0
            ws_group = []
            for id_ws in range(len(list_ws)):
                x1_ws, y1_ws, x2_ws, y2_ws = list_ws[id_ws]

                # Update metrics
                nb_ws += 1
                tot_height_ws += y2_ws - y1_ws
                min_height_ws, max_height_ws = min(y1_ws, min_height_ws), max(y2_ws, max_height_ws)
                ws_group.append([int(x_min), int(y1_ws), int(x_max), int(y2_ws)])

            # Check group relevance
            if ((tot_height_ws >= total_height)
                    and (tot_height_ws >= 0.8 * (max_height_ws - min_height_ws))
                    and ((nb_ws == 1) or (x_max - x_min >= 2 * min_width))
            ):
                final_whitespaces.append(ws_group)

    # Deduplicate whitespaces in case of continuous
    if continuous:
        dedup_whitespaces = []

        x1_prev, y1_prev, x2_prev, y2_prev = 0, 0, 0, 0
        for idx in range(len(final_whitespaces)):
            x1, y1, x2, y2 = final_whitespaces[idx][0]

            if x1 == x2_prev and y1 == y1_prev and y2 == y2_prev:
                # Merge with previous whitespace
                x2_prev = x2
            else:
                # Add whitespace
                if x2_prev - x1_prev >= min_width and idx > 0:
                    dedup_whitespaces.append([[x1_prev, y1_prev, x2_prev, y2_prev]])
                # Reset metrics
                x1_prev, y1_prev, x2_prev, y2_prev = x1, y1, x2, y2
        # Add last whitespace
        if x2_prev - x1_prev >= min_width:
            dedup_whitespaces.append([[x1_prev, y1_prev, x2_prev, y2_prev]])

        return dedup_whitespaces

    return final_whitespaces


def get_whitespaces(segment: Union[ImageSegment, ColumnGroup], vertical: bool = True, min_width: float = 0,
                    min_height: float = 1, pct: float = 0.25, continuous: bool = True) -> list[Whitespace]:
    """
    Identify whitespaces in segment
    :param segment: image segment
    :param vertical: boolean indicating if vertical or horizontal whitespaces are identified
    :param min_width: minimum width of the detected whitespaces
    :param min_height: minimum height of the detected whitespaces
    :param pct: minimum percentage of the segment height/width to account for a whitespace
    :param continuous: boolean indicating if only continuous whitespaces are retrieved
    :return: list of vertical or horizontal whitespaces
    """
    # Flip object coordinates in horizontal case
    if not vertical:
        flipped_elements = [Cell(x1=el.y1, y1=el.x1, x2=el.y2, y2=el.x2) for el in segment.elements]
        segment = ImageSegment(x1=segment.y1,
                               y1=segment.x1,
                               x2=segment.y2,
                               y2=segment.x2,
                               elements=flipped_elements)

    # Get min/max height of elements in segment
    y_min, y_max = min([el.y1 for el in segment.elements]), max([el.y2 for el in segment.elements])

    # Create array containing elements
    elements_array = np.array([[el.x1, el.y1, el.x2, el.y2] for el in segment.elements]
                              + [[segment.x1, y, segment.x2, y] for y in [y_min, y_max]])
    elements_array = np.c_[elements_array, (elements_array[:, 1] + elements_array[:, 3]) / 2]
    elements_array = elements_array[elements_array[:, 4].argsort()]

    # Compute whitespace groups
    ws_groups = compute_whitespaces(elements_array=elements_array,
                                    min_width=min_width,
                                    min_height=min_height,
                                    total_height=pct * (y_max - y_min),
                                    continuous=continuous)

    # Map to whitespaces
    whitespaces = [Whitespace(cells=[Cell(x1=c[0], y1=c[1], x2=c[2], y2=c[3]) for c in ws_gp])
                   for ws_gp in ws_groups]

    # Flip object coordinates in horizontal case
    if not vertical:
        whitespaces = [ws.flipped() for ws in whitespaces]

    return whitespaces


def adjacent_whitespaces(w_1: Whitespace, w_2: Whitespace) -> bool:
    """
    Identify if two whitespaces are adjacent
    :param w_1: first whitespace
    :param w_2: second whitespace
    :return: boolean indicating if two whitespaces are adjacent
    """
    x_coherent = len({w_1.x1, w_1.x2}.intersection({w_2.x1, w_2.x2})) > 0
    y_coherent = min(w_1.y2, w_2.y2) - max(w_1.y1, w_2.y1) > 0

    return x_coherent and y_coherent


def identify_coherent_v_whitespaces(v_whitespaces: list[Whitespace]) -> list[Whitespace]:
    """
    From vertical whitespaces, identify the most relevant ones according to height, width and relative positions
    :param v_whitespaces: list of vertical whitespaces
    :return: list of relevant vertical delimiters
    """
    deleted_idx = []
    for i in range(len(v_whitespaces)):
        for j in range(i, len(v_whitespaces)):
            # Check if both whitespaces are adjacent
            adjacent = adjacent_whitespaces(v_whitespaces[i], v_whitespaces[j])

            if adjacent:
                if v_whitespaces[i].height > v_whitespaces[j].height:
                    deleted_idx.append(j)
                elif v_whitespaces[i].height < v_whitespaces[j].height:
                    deleted_idx.append(i)

    return [ws for idx, ws in enumerate(v_whitespaces) if idx not in deleted_idx]


def deduplicate_whitespaces(ws: list[Whitespace], elements: list[Cell]) -> list[Whitespace]:
    """
    Remove useless whitespaces
    :param ws: list of whitespaces
    :param elements: list of segment elements
    :return: filtered whitespaces
    """
    if len(ws) <= 1:
        return ws

    deleted_idx, merged_ws = [], []
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            matching_elements = []
            for ws_1 in ws[i].cells:
                for ws_2 in ws[j].cells:
                    if min(ws_1.y2, ws_2.y2) - max(ws_1.y1, ws_2.y1) <= 0:
                        continue

                    # Get common area
                    common_area = Cell(x1=min(ws_1.x2, ws_2.x2),
                                       y1=max(ws_1.y1, ws_2.y1),
                                       x2=max(ws_1.x1, ws_2.x1),
                                       y2=min(ws_1.y2, ws_2.y2))

                    # Identify matching elements
                    matching_elements += [el for el in elements
                                          if min(el.x2, common_area.x2) - max(el.x1, common_area.x1) > 0
                                          and min(el.y2, common_area.y2) - max(el.y1, common_area.y1) > 0]

            if len(matching_elements) == 0:
                # Add smallest element to deleted ws
                if ws[i].height > ws[j].height:
                    deleted_idx.append(j)
                elif ws[i].height < ws[j].height:
                    deleted_idx.append(i)
                else:
                    # Create a merged whitespace
                    new_cells = [Cell(x1=min(ws[i].x1, ws[j].x1),
                                      y1=c.y1,
                                      x2=max(ws[i].x2, ws[j].x2),
                                      y2=c.y2)
                                 for c in ws[i].cells + ws[j].cells]
                    merged_ws.append(Whitespace(cells=list(set(new_cells))))
                    deleted_idx += [i, j]

    filtered_ws = [w for idx, w in enumerate(ws) if idx not in deleted_idx]

    # Remove merged whitespaces that are incoherent with filtered whitespaces
    merged_ws = [m_ws for m_ws in merged_ws
                 if not any(min(w.x2, m_ws.x2) - max(w.x1, m_ws.x1) > 0 for w in filtered_ws)]

    if len(merged_ws) > 1:
        # Deduplicate overlapping merged ws
        seq = iter(sorted(merged_ws, key=lambda w: w.area, reverse=True))
        filtered_merged_ws = [next(seq)]
        for w in seq:
            if not any(f_ws for f_ws in filtered_ws if w in f_ws):
                filtered_merged_ws.append(w)
    else:
        filtered_merged_ws = merged_ws

    return filtered_ws + filtered_merged_ws


def get_relevant_vertical_whitespaces(segment: Union[ImageSegment, ColumnGroup], char_length: float,
                                      median_line_sep: float, pct: float = 0.25) -> list[Whitespace]:
    """
    Identify vertical whitespaces that can be column delimiters
    :param segment: image segment
    :param char_length: average character width in image
    :param median_line_sep: median row separation
    :param pct: minimum percentage of the segment height for a vertical whitespace
    :return: list of vertical whitespaces that can be column delimiters
    """
    # Identify vertical whitespaces
    v_whitespaces = get_whitespaces(segment=segment,
                                    vertical=True,
                                    pct=pct,
                                    min_width=char_length,
                                    min_height=min(median_line_sep, segment.element_height),
                                    continuous=True)

    # Identify relevant vertical whitespaces that can be column delimiters
    vertical_delims = identify_coherent_v_whitespaces(v_whitespaces=v_whitespaces)

    return deduplicate_whitespaces(ws=vertical_delims, elements=segment.elements)
