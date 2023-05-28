# coding: utf-8

from typing import List

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import ImageSegment


def get_vertical_whitespaces(segment: ImageSegment) -> List[Cell]:
    """
    Identify vertical whitespaces in segment
    :param segment: image segment
    :return: list of vertical whitespaces as Cell objects
    """
    # Get min/max height of elements in segment
    y_min, y_max = min([el.y1 for el in segment.elements]), max([el.y2 for el in segment.elements])

    # Get all x values for segment elements
    x_vals = [el.x1 for el in segment.elements] + [el.x2 for el in segment.elements] + [segment.x1, segment.x2]
    x_vals = sorted(list(set(x_vals)))

    # Identify vertical whitespaces
    v_whitespaces = list()
    for x_min, x_max in zip(x_vals, x_vals[1:]):
        # Identify elements in this range
        rng_elements = sorted([el for el in segment.elements if min(el.x2, x_max) - max(el.x1, x_min) > 0],
                              key=lambda el: el.y1 + el.y2)

        if rng_elements:
            # Check top and bottom gaps
            if rng_elements[0].y1 - segment.y1 >= 0.25 * segment.height:
                v_whitespaces.append(Cell(x1=x_min, y1=segment.y1, x2=x_max, y2=rng_elements[0].y1))
            if segment.y2 - rng_elements[-1].y2 >= 0.25 * segment.height:
                v_whitespaces.append(Cell(x1=x_min, y1=rng_elements[-1].y2, x2=x_max, y2=segment.y2))

            # Check middle gaps
            for el_top, el_bottom in zip(rng_elements, rng_elements[1:]):
                if el_bottom.y1 - el_top.y2 >= 0.25 * segment.height:
                    v_whitespaces.append(Cell(x1=x_min, y1=el_top.y2, x2=x_max, y2=el_bottom.y1))
        else:
            v_whitespaces.append(Cell(x1=x_min, y1=segment.y1, x2=x_max, y2=segment.y2))

    # Merge consecutive corresponding whitespaces
    v_whitespaces = sorted(v_whitespaces, key=lambda w: (w.y1, w.y2, w.x1))

    seq = iter(v_whitespaces)
    merged_v_whitespaces = [[next(seq)]]
    for w in seq:
        prev = merged_v_whitespaces[-1][-1]
        if w.x1 != prev.x2 or not (w.y1 == prev.y1 and w.y2 == prev.y2):
            merged_v_whitespaces.append([])
        merged_v_whitespaces[-1].append(w)

    merged_v_whitespaces = [Cell(x1=min([w.x1 for w in cl]),
                                 y1=max(min([w.y1 for w in cl]), y_min),
                                 x2=max([w.x2 for w in cl]),
                                 y2=min(max([w.y2 for w in cl]), y_max))
                            for cl in merged_v_whitespaces]

    return merged_v_whitespaces


def adjacent_whitespaces(w_1: Cell, w_2: Cell) -> bool:
    """
    Identify if two whitespaces are adjacent
    :param w_1: first whitespace
    :param w_2: second whitespace
    :return: boolean indicating if two whitespaces are adjacent
    """
    x_coherent = len({w_1.x1, w_1.x2}.intersection({w_2.x1, w_2.x2})) > 0
    y_coherent = min(w_1.y2, w_2.y2) - max(w_1.y1, w_2.y1) > 0

    return x_coherent and y_coherent


def identify_coherent_v_whitespaces(v_whitespaces: List[Cell], char_length: float) -> List[Cell]:
    """
    From vertical whitespaces, identify the most relevant ones according to height, width and relative positions
    :param v_whitespaces: list of vertical whitespaces
    :param char_length: average character width in image
    :return: list of relevant vertical delimiters
    """
    # Create vertical delimiters groups
    v_groups = cluster_items(items=v_whitespaces,
                             clustering_func=adjacent_whitespaces)

    # Keep only delimiters that represent at least 75% of the height of their group
    v_delims = [d for gp in v_groups
                for d in [d for d in gp if d.height >= 0.75 * max([d.height for d in gp])]]

    # Group once again delimiters and keep only highest one in group
    v_delim_groups = cluster_items(items=v_delims,
                                   clustering_func=adjacent_whitespaces)

    # Filter delimiters in each group by width
    v_delim_groups = [[d for d in gp if d.width >= 0.5 * char_length] for gp in v_delim_groups]

    # For each group, select a delimiter that has the largest height
    final_delims = list()
    for gp in v_delim_groups:
        if gp:
            # Get x center of group
            x_center = (min([d.x1 for d in gp]) + max([d.x2 for d in gp]))

            # Filter on tallest delimiters
            tallest_delimiters = [d for d in gp if d.height == max([d.height for d in gp])]

            # Add delimiter closest to the center of the group
            closest_del = sorted(tallest_delimiters, key=lambda d: abs(d.x1 + d.x2 - x_center)).pop(0)
            final_delims.append(closest_del)

    return final_delims


def get_relevant_vertical_whitespaces(segment: ImageSegment, char_length: float) -> List[Cell]:
    """
    Identify vertical whitespaces that can be column delimiters
    :param segment: image segment
    :param char_length: average character width in image
    :return: list of vertical whitespaces that can be column delimiters
    """
    # Identify vertical whitespaces
    v_whitespaces = get_vertical_whitespaces(segment=segment)

    # Identify relevant vertical whitespaces that can be column delimiters
    vertical_delims = identify_coherent_v_whitespaces(v_whitespaces=v_whitespaces,
                                                      char_length=char_length)

    return vertical_delims
