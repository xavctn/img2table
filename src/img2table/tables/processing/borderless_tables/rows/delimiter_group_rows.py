# coding: utf-8
from typing import List

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup
from img2table.tables.processing.borderless_tables.whitespaces import get_whitespaces


def identify_row_delimiters(delimiter_group: DelimiterGroup) -> List[Cell]:
    """
    Identify list of rows corresponding to the delimiter group
    :param delimiter_group: column delimiters group
    :return: list of rows delimiters corresponding to the delimiter group
    """
    # Identify vertical whitespaces
    h_ws = get_whitespaces(segment=delimiter_group, vertical=False, pct=0.66)

    # Create horizontal delimiters groups
    h_groups = cluster_items(items=h_ws,
                             clustering_func=lambda w1, w2: len({w1.y1, w1.y2}.intersection({w2.y1, w2.y2})) > 0)

    # For each group, select only delimiters that have the largest width
    final_delims = list()
    for gp in h_groups:
        for delim in gp:
            if delim.y1 == delimiter_group.y1 or delim.y2 == delimiter_group.y2:
                continue

            # Get adjacent delimiters
            adjacent_delims = [d for d in gp if d != delim and len({delim.y1, delim.y2}.intersection({d.y1, d.y2})) > 0]

            if len(adjacent_delims) == 0:
                final_delims.append(Cell(x1=delim.x1,
                                         x2=delim.x2,
                                         y1=(delim.y1 + delim.y2) // 2,
                                         y2=(delim.y1 + delim.y2) // 2))
            elif delim.width >= max([d.width for d in adjacent_delims]):
                final_delims.append(Cell(x1=delim.x1,
                                         x2=delim.x2,
                                         y1=(delim.y1 + delim.y2) // 2,
                                         y2=(delim.y1 + delim.y2) // 2))

    final_delims += [Cell(x1=delimiter_group.x1, x2=delimiter_group.x2, y1=delimiter_group.y1, y2=delimiter_group.y1),
                     Cell(x1=delimiter_group.x1, x2=delimiter_group.x2, y1=delimiter_group.y2, y2=delimiter_group.y2)]

    return sorted(final_delims, key=lambda d: d.y1)


def filter_coherent_row_delimiters(row_delimiters: List[Cell], delimiter_group: DelimiterGroup) -> List[Cell]:
    """
    Filter coherent row delimiters (i.e that properly delimit relevant text)
    :param row_delimiters: list of row delimiters
    :param delimiter_group: column delimiters group
    :return: filtered row delimiters
    """
    # Get max width of delimiters
    max_width = max(map(lambda d: d.width, row_delimiters))

    delimiters_to_delete = list()
    for idx, delim in enumerate(row_delimiters):
        if delim.width >= 0.95 * max_width:
            continue

        # Get area above delimiter and corresponding columns
        upper_delim = row_delimiters[idx - 1]
        upper_area = Cell(x1=max(delim.x1, upper_delim.x1),
                          y1=upper_delim.y2,
                          x2=min(delim.x2, upper_delim.x2),
                          y2=delim.y1)
        upper_columns = sorted([col for col in delimiter_group.delimiters
                                if min(upper_area.y2, col.y2) - max(upper_area.y1, col.y1) >= 0.8 * upper_area.height
                                and upper_area.x1 <= col.x1 <= upper_area.x2],
                               key=lambda c: c.x1)
        # Get contained elements in upper area
        upper_contained_elements = [el for el in delimiter_group.elements if el.y1 >= upper_area.y1
                                    and el.y2 <= upper_area.y2 and el.x1 >= upper_columns[0].x2
                                    and el.x2 <= upper_columns[-1].x1] if upper_columns else []

        # Get area below delimiter and corresponding columns
        bottom_delim = row_delimiters[idx + 1]
        bottom_area = Cell(x1=max(delim.x1, bottom_delim.x1),
                           y1=delim.y2,
                           x2=min(delim.x2, bottom_delim.x2),
                           y2=bottom_delim.y1)
        bottom_columns = sorted([col for col in delimiter_group.delimiters
                                 if min(bottom_area.y2, col.y2) - max(bottom_area.y1, col.y1) >= 0.8 * bottom_area.height
                                 and bottom_area.x1 <= col.x1 <= bottom_area.x2],
                                key=lambda c: c.x1)
        # Get contained elements in bottom area
        bottom_contained_elements = [el for el in delimiter_group.elements if el.y1 >= bottom_area.y1
                                     and el.y2 <= bottom_area.y2 and el.x1 >= bottom_columns[0].x2
                                     and el.x2 <= bottom_columns[-1].x1] if bottom_columns else []

        # If one of the area is empty, the delimiter is irrelevant
        if len(upper_contained_elements) * len(bottom_contained_elements) == 0:
            delimiters_to_delete.append(idx)

    return [d for idx, d in enumerate(row_delimiters) if idx not in delimiters_to_delete]


def correct_delimiter_width(row_delimiters: List[Cell], contours: List[Cell]) -> List[Cell]:
    """
    Correct delimiter width if needed
    :param row_delimiters: list of row delimiters
    :param contours: list of image contours
    :return: list of row delimiters with corrected width
    """
    x_min, x_max = min([d.x1 for d in row_delimiters]), max([d.x2 for d in row_delimiters])

    for idx, delim in enumerate(row_delimiters):
        if delim.width == x_max - x_min:
            continue

        # Check if there are contours on the left of the delimiter
        left_contours = [c for c in contours if c.y1 + c.height // 6 < delim.y1 < c.y2 - c.height // 6
                         and min(c.x2, delim.x1) - max(c.x1, x_min) > 0]
        delim_x_min = max([c.x2 for c in left_contours] + [x_min])

        # Check if there are contours on the right of the delimiter
        right_contours = [c for c in contours if c.y1 + c.height // 6 < delim.y1 < c.y2 - c.height // 6
                          and min(c.x2, x_max) - max(c.x1, delim.x2) > 0]
        delim_x_max = min([c.x1 for c in right_contours] + [x_max])

        # Update delimiter width
        setattr(row_delimiters[idx], "x1", delim_x_min)
        setattr(row_delimiters[idx], "x2", delim_x_max)

    return row_delimiters


def identify_delimiter_group_rows(delimiter_group: DelimiterGroup, contours: List[Cell]) -> List[Cell]:
    """
    Identify list of rows corresponding to the delimiter group
    :param delimiter_group: column delimiters group
    :param contours: list of image contours
    :return: list of rows delimiters corresponding to the delimiter group
    """
    # Get row delimiters
    row_delimiters = identify_row_delimiters(delimiter_group=delimiter_group)

    if row_delimiters:
        # Filter coherent delimiters
        coherent_delimiters = filter_coherent_row_delimiters(row_delimiters=row_delimiters,
                                                             delimiter_group=delimiter_group)

        # Correct delimiters width
        corrected_delimiters = correct_delimiter_width(row_delimiters=coherent_delimiters,
                                                       contours=contours)

        return corrected_delimiters if len(corrected_delimiters) >= 3 else []
    return []

