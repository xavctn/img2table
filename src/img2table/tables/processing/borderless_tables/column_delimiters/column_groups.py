# coding: utf-8

from typing import List

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, ImageSegment
from img2table.tables.processing.common import is_contained_cell


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


def get_coherent_height(delimiter_group: DelimiterGroup, segment: ImageSegment,
                        delimiters: List[Cell]) -> DelimiterGroup:
    """
    Identify height for a delimiter group based on position of elements within the image
    :param delimiter_group: delimiter group
    :param segment: Image segment object
    :param delimiters: list of vertical delimiters as Cell objects
    :return: processed delimiter group
    """
    # Get elements that correspond to the delimiter group
    delim_elements = [el for el in segment.elements
                      if is_contained_cell(inner_cell=el, outer_cell=delimiter_group.bbox)]

    # Group elements in rows
    seq = iter(sorted(delim_elements, key=lambda el: (el.y1, el.y2)))
    lines = [[next(seq)]]
    for el in seq:
        y2_line = max([el.y2 for el in lines[-1]])
        if el.y1 > y2_line:
            lines.append([])
        lines[-1].append(el)

    # Identify top value for delimiters and elements in delimiter group
    y_top, y_bottom, delim_group_elements = delimiter_group.bbox.y2, delimiter_group.bbox.y1, []
    for line in lines:
        x1_line, x2_line = min([el.x1 for el in line]), max([el.x2 for el in line])
        y1_line, y2_line = min([el.y1 for el in line]), max([el.y2 for el in line])

        # Identify delimiters that correspond vertically to rows
        line_delims = [d for d in delimiter_group.delimiters if min(d.y2, y2_line) - max(d.y1, y1_line) > 0]

        if len([d for d in line_delims if min(d.x2, x2_line) - max(d.x1, x1_line) > 0]) > 0:
            y_top = min(y_top, y1_line)
            y_bottom = max(y_bottom, y2_line)

    # Get elements corresponding to found area
    delim_group_elements = [el for line in lines for el in line
                            if min([el.y1 for el in line]) >= y_top and max([el.y2 for el in line]) <= y_bottom]

    # Reprocess delimiters
    processed_delimiters = [Cell(x1=d.x1,
                                 x2=d.x2,
                                 y1=max(d.y1, y_top),
                                 y2=min(d.y2, y_bottom))
                            for d in delimiter_group.delimiters]
    processed_delimiters = [d for d in processed_delimiters if d.height >= 0.75 * (y_bottom - y_top)]

    # Create delimiter group
    delimiter_group = DelimiterGroup(delimiters=processed_delimiters,
                                     elements=delim_group_elements)

    # Identify other delimiters that could match
    other_delimiters = [Cell(x1=d.x1, y1=max(d.y1, delimiter_group.y1), x2=d.x2, y2=min(d.y2, delimiter_group.y2))
                        for d in delimiters
                        if d.x1 > delimiter_group.x1 and d.x2 < delimiter_group.x2
                        and min(d.y2, delimiter_group.y2) - max(d.y1, delimiter_group.y1) >= 0.75 * delimiter_group.height]

    for delim in other_delimiters:
        if delim not in delimiter_group.delimiters:
            delimiter_group.add(delim)

    return delimiter_group


def create_delimiter_groups(delimiters: List[Cell], segment: ImageSegment) -> List[DelimiterGroup]:
    """
    Identify groups of vertical delimiters that can correspond to table columns
    :param delimiters: list of vertical delimiters as Cell objects
    :param segment: Image segment object
    :return: list of delimiter groups that can correspond to columns
    """
    # Cluster delimiters into several groups
    delimiter_groups = group_delimiters(delimiters=delimiters)

    # Deduplicate delimiter groups
    deduplicated_groups = deduplicate_groups(delimiter_groups=delimiter_groups)

    # Reprocess delimiter groups height
    processed_delim_groups = [get_coherent_height(delimiter_group=gp,
                                                  segment=segment,
                                                  delimiters=delimiters)
                              for gp in deduplicated_groups]

    return processed_delim_groups
