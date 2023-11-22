# coding: utf-8
from functools import partial
from typing import List, Tuple

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import TableSegment, DelimiterGroup


def get_coherent_ws_height(vertical_ws: List[Cell], unused_ws: List[Cell],
                           elements: List[Cell]) -> Tuple[List[Cell], List[Cell]]:
    """
    Get whitespaces with coherent height in relationship with elements
    :param vertical_ws: vertical whitespaces from segment
    :param unused_ws: list of unused whitespaces
    :param elements: elements in segment
    :return: tuple containing list of vertical whitespaces and list of unused whitespaces resized
    """
    # Define relevant ws
    relevant_ws = [ws for ws in unused_ws if ws.height >= 0.66 * max([w.height for w in vertical_ws])]
    relevant_ws += vertical_ws

    # Group elements in rows
    seq = iter(sorted(elements, key=lambda el: (el.y1, el.y2)))
    rows = [[next(seq)]]
    for el in seq:
        y2_row = max([el.y2 for el in rows[-1]])
        if el.y1 >= y2_row:
            rows.append([])
        rows[-1].append(el)
    
    # Identify top and bottom values for vertical whitespaces
    y_top, y_bottom, = max([ws.y2 for ws in relevant_ws]), min([ws.y1 for ws in relevant_ws])
    for row in rows:
        x1_row, x2_row = min([el.x1 for el in row]), max([el.x2 for el in row])
        y1_row, y2_row = min([el.y1 for el in row]), max([el.y2 for el in row])

        # Identify whitespaces that correspond vertically to rows
        row_ws = [ws for ws in relevant_ws
                  if min(ws.y2, y2_row) - max(ws.y1, y1_row) == y2_row - y1_row]

        if len([ws for ws in row_ws if min(ws.x2, x2_row) - max(ws.x1, x1_row) > 0]) > 0:
            y_top = min(y_top, y1_row)
            y_bottom = max(y_bottom, y2_row)

    # Reprocess whitespaces
    vertical_ws = [Cell(x1=ws.x1, y1=max(ws.y1, y_top), x2=ws.x2, y2=min(ws.y2, y_bottom)) for ws in vertical_ws]
    unused_ws = [Cell(x1=ws.x1, y1=max(ws.y1, y_top), x2=ws.x2, y2=min(ws.y2, y_bottom)) for ws in unused_ws]

    return vertical_ws, unused_ws


def corresponding_whitespaces(ws_1: Cell, ws_2: Cell, char_length: float, median_line_sep: float) -> bool:
    """
    Identify if whitespaces can correspond vertically
    :param ws_1: first whitespace
    :param ws_2: second whitespace
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: boolean indicating if whitespaces can correspond vertically
    """
    if min(abs(ws_2.y2 - ws_1.y1), abs(ws_1.y2 - ws_2.y1),
           abs(ws_1.y1 - ws_2.y1), abs(ws_2.y2 - ws_1.y2)) > 2 * median_line_sep:
        return False

    return min(ws_1.x2, ws_2.x2) - max(ws_1.x1, ws_2.x1) >= -char_length / 2


def identify_missing_vertical_whitespaces(unused_ws: List[Cell], char_length: float, median_line_sep: float,
                                          ref_height: int) -> List[Cell]:
    """
    Identify potential missing delimiters
    :param unused_ws: list of unused whitespace
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param ref_height: reference height
    :return: list of newly created whitespaces
    """
    # Create clusters of corresponding whitespaces
    f_cluster = partial(corresponding_whitespaces, char_length=char_length, median_line_sep=median_line_sep)
    ws_clusters = cluster_items(items=unused_ws,
                                clustering_func=f_cluster)

    new_ws = list()
    # Check if clusters can create a new vertical whitespace
    for cl in ws_clusters:
        if max([ws.y2 for ws in cl]) - min([ws.y1 for ws in cl]) >= 0.66 * ref_height:
            v_ws = Cell(x1=min([ws.x1 for ws in cl]),
                        y1=min([ws.y1 for ws in cl]),
                        x2=max([ws.x2 for ws in cl]),
                        y2=max([ws.y2 for ws in cl]))
            new_ws.append(v_ws)

    return new_ws


def distance_to_elements(x: int, elements: List[Cell]) -> Tuple[int, float]:
    """
    Compute distance metrics of elements to an x value
    :param x: x value
    :param elements: elements
    :return: distance / number of avoided elements
    """
    distance_elements = 0
    number_avoided_elements = 0
    for el in elements:
        if el.x1 <= x <= el.x2:
            distance_elements -= min(abs(el.x1 - x), abs(el.x2 - x)) ** 1 / 3
        else:
            number_avoided_elements += 1
            distance_elements += min(abs(el.x1 - x), abs(el.x2 - x)) ** 1 / 3

    return number_avoided_elements, distance_elements


def get_coherent_whitespace_position(ws: Cell, elements: List[Cell]) -> Cell:
    """
    Get coherent whitespace position for whitespace in relationship to segment elements
    :param ws: whitespace
    :param elements: segment elements
    :return: final whitespace
    """
    # Get potential conflicting elements
    conflicting_els = [el for el in elements if min(el.x2, ws.x2) - max(el.x1, ws.x1) > 0
                       and min(el.y2, ws.y2) - max(el.y1, ws.y1) > 0]

    if conflicting_els:
        # Get x value that maximises the distance to conflicting elements
        x_ws = sorted(range(ws.x1, ws.x2 + 1),
                      key=lambda x: distance_to_elements(x, conflicting_els),
                      reverse=True)[0]
    else:
        # Get elements to left and right of ws
        left_els = [el for el in elements if min(el.y2, ws.y2) - max(el.y1, ws.y1) > 0 and el.x2 <= ws.x1]
        right_els = [el for el in elements if min(el.y2, ws.y2) - max(el.y1, ws.y1) > 0 and ws.x2 <= el.x1]

        if len(left_els) > 0 and len(right_els) > 0:
            x_ws = round((max([el.x2 for el in left_els]) + min([el.x1 for el in right_els])) / 2)
        elif len(left_els) > 0:
            x_ws = max([el.x2 for el in left_els])
        elif len(right_els) > 0:
            x_ws = min([el.x1 for el in right_els])
        else:
            x_ws = round((ws.x1 + ws.x2) / 2)

    return Cell(x1=x_ws, y1=ws.y1, x2=x_ws, y2=ws.y2)


def filter_coherent_delimiters(delimiters: List[Cell], elements: List[Cell]) -> List[Cell]:
    # Check delimiters coherency (i.e) if it adds value
    filtered_delims = list()
    for delim in delimiters:
        left_delims = sorted([d for d in delimiters if d != delim and d.x2 < delim.x1
                              and min(delim.y2, d.y2) - max(delim.y1, d.y1) > 0
                              and d.height >= delim.height],
                             key=lambda d: d.x2)
        right_delims = sorted([d for d in delimiters if d != delim and d.x1 > delim.x2
                               and min(delim.y2, d.y2) - max(delim.y1, d.y1) > 0
                               and d.height >= delim.height],
                              key=lambda d: d.x1,
                              reverse=True)
        if len(right_delims) > 0 and len(left_delims) > 0:
            left_delim, right_delim = left_delims.pop(), right_delims.pop()
            # Get elements between delimiters
            left_els = [el for el in elements if el.x1 >= left_delim.x2 and el.x2 <= delim.x1
                        and min(el.y2, min(delim.y2, left_delim.y2)) - max(el.y1, max(delim.y1, left_delim.y1)) > 0]
            right_els = [el for el in elements if el.x1 >= delim.x2 and el.x2 <= right_delim.x1
                         and min(el.y2, min(delim.y2, right_delim.y2)) - max(el.y1, max(delim.y1, right_delim.y1)) > 0]
            if len(left_els) * len(right_els) > 0:
                filtered_delims.append(delim)
        elif len(right_delims) > 0:
            right_delim = right_delims.pop()
            # Get elements between delimiters
            right_els = [el for el in elements if el.x1 >= delim.x2 and el.x2 <= right_delim.x1
                         and min(el.y2, min(delim.y2, right_delim.y2)) - max(el.y1, max(delim.y1, right_delim.y1)) > 0]
            if len(right_els) > 0:
                filtered_delims.append(delim)
        elif len(left_delims) > 0:
            left_delim = left_delims.pop()
            # Get elements between delimiters
            left_els = [el for el in elements if el.x1 >= left_delim.x2 and el.x2 <= delim.x1
                        and min(el.y2, min(delim.y2, left_delim.y2)) - max(el.y1, max(delim.y1, left_delim.y1)) > 0]
            if len(left_els) > 0:
                filtered_delims.append(delim)
        else:
            filtered_delims.append(delim)

    return filtered_delims


def get_column_whitespaces(vertical_ws: List[Cell], unused_ws: List[Cell],
                           table_segment: TableSegment, char_length: float, median_line_sep: float) -> DelimiterGroup:
    """
    Identify all whitespaces that can be used as column delimiters in the table segment
    :param vertical_ws: list of vertical whitespaces in table segment
    :param unused_ws: list of unused whitespaces in table segment
    :param table_segment: table segment
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: delimiter group
    """
    # Get whitespaces with coherent height in relationship with elements
    vertical_ws, unused_ws = get_coherent_ws_height(vertical_ws=vertical_ws,
                                                    unused_ws=unused_ws,
                                                    elements=table_segment.elements)

    # Identify potential missing delimiters
    ref_height = max([ws.y2 for ws in vertical_ws]) - min([ws.y1 for ws in vertical_ws])
    missing_ws = identify_missing_vertical_whitespaces(unused_ws=unused_ws,
                                                       char_length=char_length,
                                                       median_line_sep=median_line_sep,
                                                       ref_height=ref_height)

    # Get final delimiters positions
    final_delims = list(set([get_coherent_whitespace_position(ws=ws,
                                                              elements=table_segment.elements)
                             for ws in vertical_ws + missing_ws]))

    # Filtered useful delimiters
    useful_delims = filter_coherent_delimiters(delimiters=final_delims,
                                               elements=table_segment.elements)

    # Create delimiter group
    x1_del, x2_del = min([d.x1 for d in useful_delims]), max([d.x2 for d in useful_delims])
    y1_del, y2_del = min([d.y1 for d in useful_delims]), max([d.y2 for d in useful_delims])
    delimiter_group = DelimiterGroup(delimiters=useful_delims,
                                     elements=[el for el in table_segment.elements if el.x1 >= x1_del
                                               and el.x2 <= x2_del and el.y1 >= y1_del and el.y2 <= y2_del])

    return delimiter_group
