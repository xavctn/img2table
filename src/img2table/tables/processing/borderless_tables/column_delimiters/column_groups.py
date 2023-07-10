# coding: utf-8

from typing import List

import numpy as np

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, ImageSegment


def vertically_coherent_delimiters(d_1: Cell, d_2: Cell) -> bool:
    """
    Identify if the two vertical delimiters are vertically coherent
    :param d_1: first vertical delimiter
    :param d_2: second vertical delimiter
    :return: boolean indicating if the two vertical delimiters are vertically coherent
    """
    y_common = min(d_1.y2, d_2.y2) - max(d_1.y1, d_2.y1)
    return y_common >= 0.5 * max(d_1.height, d_2.height)


def group_delimiters(delimiters: List[Cell]) -> List[DelimiterGroup]:
    """
    Create vertical delimiter groups based on vertical positions
    :param delimiters: list of vertical delimiters as Cell objects
    :return: list of delimiter groups
    """
    # Create vertical delimiters groups
    vertical_groups = cluster_items(items=delimiters,
                                    clustering_func=vertically_coherent_delimiters)

    return [DelimiterGroup(delimiters=v_group) for v_group in vertical_groups if len(v_group) >= 4]


def deduplicate_groups(delimiter_groups: List[DelimiterGroup]) -> List[DelimiterGroup]:
    """
    Deduplicate and remove groups of delimiters that are intertwined between each other
    :param delimiter_groups: list of delimiter groups
    :return: deduplicated list of delimiter groups
    """
    dedup_delimiter_groups = list()
    for idx, delim_gp in enumerate(delimiter_groups):
        # Get other groups
        other_gps = delimiter_groups[:idx] + delimiter_groups[idx + 1:]

        # Find groups with matching positions
        matching_groups = [gp for gp in other_gps
                           if min(delim_gp.x2, gp.x2) - max(delim_gp.x1, gp.x1) > 0
                           and min(delim_gp.y2, gp.y2) - max(delim_gp.y1, gp.y1) > 0]

        # Identify groups that have delimiters that are within the current group area
        conflicting_delims = [d for gp in matching_groups for d in gp.delimiters
                              if d.x1 >= delim_gp.x1 and d.x2 <= delim_gp.x2
                              and min(delim_gp.y2, d.y2) - max(delim_gp.y1, d.y1) > 0]

        if len(conflicting_delims) == 0:
            # If there is no conflicting delimiters, append to deduplicated list
            dedup_delimiter_groups.append(delim_gp)

    return dedup_delimiter_groups


def get_coherent_height(delimiter_group: DelimiterGroup, segment: ImageSegment) -> DelimiterGroup:
    """
    Identify height for a delimiter group based on position of elements within the image
    :param delimiter_group: delimiter group
    :param segment: Image segment object
    :return: processed delimiter group
    """
    # Get elements that correspond to the delimiter group
    delim_elements = [el for el in segment.elements
                      if el.y1 >= delimiter_group.y1 and el.y2 <= delimiter_group.y2
                      and el.x1 >= min([d.x2 for d in delimiter_group.delimiters])
                      and el.x2 <= max([d.x1 for d in delimiter_group.delimiters])]

    if len(delim_elements) == 0:
        return delimiter_group

    # Group elements in rows
    seq = iter(sorted(delim_elements, key=lambda el: (el.y1, el.y2)))
    lines = [[next(seq)]]
    for el in seq:
        y2_line = max([el.y2 for el in lines[-1]])
        if el.y1 >= y2_line:
            lines.append([])
        lines[-1].append(el)

    # Identify top value for delimiters in delimiter group
    y_top, y_bottom, = delimiter_group.bbox.y2, delimiter_group.bbox.y1
    for line in lines:
        x1_line, x2_line = min([el.x1 for el in line]), max([el.x2 for el in line])
        y1_line, y2_line = min([el.y1 for el in line]), max([el.y2 for el in line])

        # Identify delimiters that correspond vertically to rows
        line_delims = [d for d in delimiter_group.delimiters
                       if min(d.y2, y2_line) - max(d.y1, y1_line) == y2_line - y1_line]

        if len([d for d in line_delims if min(d.x2, x2_line) - max(d.x1, x1_line) > 0]) > 0:
            y_top = min(y_top, y1_line)
            y_bottom = max(y_bottom, y2_line)

    # Reprocess delimiters
    processed_delimiters = [Cell(x1=d.x1,
                                 x2=d.x2,
                                 y1=max(d.y1, y_top),
                                 y2=min(d.y2, y_bottom))
                            for d in delimiter_group.delimiters]
    processed_delimiters = [d for d in processed_delimiters if d.height >= 0.75 * (y_bottom - y_top)]

    # Get corresponding elements
    delim_group_elements = [el for el in segment.elements
                            if el.y1 >= y_top and el.y2 <= y_bottom
                            and el.x1 >= min([d.x2 for d in processed_delimiters])
                            and el.x2 <= max([d.x1 for d in processed_delimiters])]

    # Create delimiter group
    delimiter_group = DelimiterGroup(delimiters=processed_delimiters,
                                     elements=delim_group_elements)

    return delimiter_group


def check_elements_vs_delimiter_group(delimiter_group: DelimiterGroup, elements: List[Cell]) -> bool:
    """
    Check if elements are coherent with an existing delimiter group
    :param delimiter_group: delimiter group
    :param elements: list of elements
    :return: boolean indicating if elements are coherent with an existing delimiter group
    """
    if len(elements) == 0:
        return False

    # For each element, check if an existing element is vertically aligned
    matching_els = list()
    for element in elements:
        # Check vertical alignment with existing elements
        y_coherent = max([abs(element.y1 + element.y2 - el.y1 - el.y2) <= 0.1 * delimiter_group.height
                          for el in delimiter_group.elements])
        matching_els.append(y_coherent)

    return np.mean(matching_els) >= 0.8


def get_complete_group(delimiter_group: DelimiterGroup, delimiters: List[Cell],
                       segment: ImageSegment) -> DelimiterGroup:
    """
    Add relevant delimiters to the group by checking intertwined and edge delimiters
    :param delimiter_group: group of delimiters
    :param delimiters: list of all delimiters
    :param segment: Image segment object
    :return: processed delimiter group
    """
    # Identify other delimiters within the group that could match
    inside_delimiters = [Cell(x1=d.x1, y1=max(d.y1, delimiter_group.y1), x2=d.x2, y2=min(d.y2, delimiter_group.y2))
                         for d in delimiters
                         if d.x1 > delimiter_group.x1 and d.x2 < delimiter_group.x2
                         and min(d.y2, delimiter_group.y2) - max(d.y1, delimiter_group.y1) >= 0.75 * delimiter_group.height]

    # Add inside delimiters to group
    for delim in inside_delimiters:
        if delim not in delimiter_group.delimiters:
            delimiter_group.add(delim)

    # Get previous left and next right delimiter
    matching_delimiters = sorted([d for d in delimiters
                                  if min(d.y2, delimiter_group.y2) - max(d.y1, delimiter_group.y1) > 0],
                                 key=lambda d: (d.x1, d.height))

    # Check left delimiter
    while len([d for d in matching_delimiters if d.x2 < delimiter_group.x1]) > 0:
        # Get delimiter
        left_delim = [d for d in matching_delimiters if d.x2 < delimiter_group.x1][-1]

        # Check if it corresponds vertically to delimiter
        y_corresponds = (min(left_delim.y2, delimiter_group.y2)
                         - max(left_delim.y1, delimiter_group.y1)) >= 0.75 * delimiter_group.height

        # Get new elements
        new_elements = [el for el in segment.elements
                        if el.y1 >= delimiter_group.y1 and el.y2 <= delimiter_group.y2
                        and left_delim.x2 <= el.x1 <= min([d.x1 for d in delimiter_group.delimiters])
                        and left_delim.x2 <= el.x2 <= min([d.x1 for d in delimiter_group.delimiters])]

        # Check elements versus delimiter group
        elements_correspond = check_elements_vs_delimiter_group(delimiter_group=delimiter_group,
                                                                elements=new_elements)

        if y_corresponds and elements_correspond:
            new_delim = Cell(x1=left_delim.x1,
                             x2=left_delim.x2,
                             y1=max(left_delim.y1, delimiter_group.y1),
                             y2=min(left_delim.y2, delimiter_group.y2))
            delimiter_group = DelimiterGroup(delimiters=delimiter_group.delimiters + [new_delim],
                                             elements=delimiter_group.elements + new_elements)
        else:
            break

    # Check right delimiter
    while len([d for d in matching_delimiters if d.x1 > delimiter_group.x2]) > 0:
        # Get delimiter
        right_delim = [d for d in matching_delimiters if d.x1 > delimiter_group.x2][0]

        # Check if it corresponds vertically to delimiter
        y_corresponds = (min(right_delim.y2, delimiter_group.y2)
                         - max(right_delim.y1, delimiter_group.y1)) >= 0.75 * delimiter_group.height

        # Get new elements
        new_elements = [el for el in segment.elements
                        if el.y1 >= delimiter_group.y1 and el.y2 <= delimiter_group.y2
                        and max([d.x2 for d in delimiter_group.delimiters]) >= el.x1 >= right_delim.x1
                        and max([d.x2 for d in delimiter_group.delimiters]) >= el.x2 >= right_delim.x1]

        # Check elements versus delimiter group
        elements_correspond = check_elements_vs_delimiter_group(delimiter_group=delimiter_group,
                                                                elements=new_elements)
        if y_corresponds and elements_correspond:
            new_delim = Cell(x1=right_delim.x1,
                             x2=right_delim.x2,
                             y1=max(right_delim.y1, delimiter_group.y1),
                             y2=min(right_delim.y2, delimiter_group.y2))
            delimiter_group = DelimiterGroup(delimiters=delimiter_group.delimiters + [new_delim],
                                             elements=delimiter_group.elements + new_elements)
        else:
            break

    return delimiter_group


def get_full_delimiters(delimiter_group: DelimiterGroup, char_length: float) -> DelimiterGroup:
    """
    Identify all relevant delimiters by selecting only vertical whitespaces within the delimiter group bbox
    :param delimiter_group: delimiter group
    :param char_length: average character width in image
    :return: delimiter group with all relevant delimiters
    """
    from img2table.tables.processing.borderless_tables.column_delimiters import get_relevant_vertical_whitespaces

    whitespaces = get_relevant_vertical_whitespaces(segment=delimiter_group,
                                                    char_length=char_length,
                                                    pct=0.75)

    return DelimiterGroup(delimiters=whitespaces,
                          elements=delimiter_group.elements)


def create_delimiter_groups(delimiters: List[Cell], segment: ImageSegment, char_length: float) -> List[DelimiterGroup]:
    """
    Identify groups of vertical delimiters that can correspond to table columns
    :param delimiters: list of vertical delimiters as Cell objects
    :param segment: Image segment object
    :param char_length: average character width in image
    :return: list of delimiter groups that can correspond to columns
    """
    # Cluster delimiters into several groups
    delimiter_groups = group_delimiters(delimiters=delimiters)

    # Deduplicate delimiter groups
    deduplicated_groups = deduplicate_groups(delimiter_groups=delimiter_groups)

    # Reprocess delimiter groups height
    processed_delim_groups = [get_coherent_height(delimiter_group=gp,
                                                  segment=segment)
                              for gp in deduplicated_groups]

    # Complete delimiter groups with other matching delimiters
    complete_delimiter_groups = [get_complete_group(delimiter_group=gp,
                                                    delimiters=delimiters,
                                                    segment=segment)
                                 for gp in processed_delim_groups if len(gp.elements) > 0]

    # Identify exhaustive list of delimiters based on only delimiter group area
    return [get_full_delimiters(delimiter_group=gp, char_length=char_length)
            for gp in complete_delimiter_groups if gp.area > 0]
