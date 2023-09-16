# coding: utf-8
from typing import List

import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import ImageSegment
from img2table.tables.processing.borderless_tables.whitespaces import get_whitespaces, \
    get_relevant_vertical_whitespaces


def get_table_areas(segment: ImageSegment, char_length: float, median_line_sep: float) -> List[ImageSegment]:
    """
    Identify relevant table areas in segment
    :param segment: ImageSegment object
    :param char_length: average character length in image
    :param median_line_sep: median line separation
    :return: list of table areas of segment
    """
    # Identify horizontal whitespaces in segment that represent at least half of median line separation
    h_ws = get_whitespaces(segment=segment, vertical=False, pct=1)
    h_ws = [ws for ws in h_ws if ws.height >= 0.5 * median_line_sep]

    # Handle case where no whitespaces have been found by creating "fake" ws at the top or bottom
    if len(h_ws) == 0:
        h_ws = [Cell(x1=min([el.x1 for el in segment.elements]),
                     x2=max([el.x2 for el in segment.elements]),
                     y1=segment.y1,
                     y2=segment.y1),
                Cell(x1=min([el.x1 for el in segment.elements]),
                     x2=max([el.x2 for el in segment.elements]),
                     y1=segment.y2,
                     y2=segment.y2)
                ]

    # Create whitespaces at the top or the bottom if they are missing
    if h_ws[0].y1 > segment.y1:
        up_ws = Cell(x1=min([ws.x1 for ws in h_ws]),
                     x2=max([ws.x2 for ws in h_ws]),
                     y1=segment.y1,
                     y2=segment.y1)
        h_ws.insert(0, up_ws)

    if h_ws[-1].y2 < segment.y2:
        down_ws = Cell(x1=min([ws.x1 for ws in h_ws]),
                       x2=max([ws.x2 for ws in h_ws]),
                       y1=segment.y2,
                       y2=segment.y2)
        h_ws.insert(-1, down_ws)

    # Check in areas between horizontal whitespaces in order to identify if they can correspond to tables
    table_areas = list()
    h_ws = sorted(h_ws, key=lambda ws: ws.y1)
    idx = 0
    for up, down in zip(h_ws, h_ws[1:]):
        idx += 1
        # Get the delimited area
        delimited_area = Cell(x1=min(up.x1, down.x1), y1=up.y2, x2=max(up.x2, down.x2), y2=down.y1)

        # Identify corresponding elements and create a corresponding segment
        area_elements = [el for el in segment.elements if el.x1 >= delimited_area.x1 and el.x2 <= delimited_area.x2
                         and el.y1 >= delimited_area.y1 and el.y2 <= delimited_area.y2]
        seg_area = ImageSegment(x1=delimited_area.x1,
                                x2=delimited_area.x2,
                                y1=delimited_area.y1,
                                y2=delimited_area.y2,
                                elements=area_elements,
                                position=idx)

        if area_elements:
            # Identify vertical whitespaces in the area
            v_ws = get_relevant_vertical_whitespaces(segment=seg_area, char_length=char_length)
            v_ws = [ws for ws in v_ws if ws.height >= 0.25 * seg_area.height and ws.area > 0]

            # Identify number of whitespaces that are not on borders
            middle_ws = [ws for ws in v_ws if ws.x1 != seg_area.x1 and ws.x2 != seg_area.x2]

            # If there can be at least 3 columns in area, it is a possible table area
            if len(middle_ws) >= 2:
                # Add edges whitespaces if they do not exist
                if not any([ws for ws in v_ws if ws.x1 == seg_area.x1]):
                    v_ws.append(Cell(x1=seg_area.x1, x2=seg_area.x1 + 1, y1=seg_area.y1, y2=seg_area.y2))
                if not any([ws for ws in v_ws if ws.x2 == seg_area.x2]):
                    v_ws.append(Cell(x1=seg_area.x2 - 1, x2=seg_area.x2, y1=seg_area.y1, y2=seg_area.y2))

                seg_area.set_whitespaces(whitespaces=v_ws)
                table_areas.append(seg_area)

    return table_areas


def coherent_table_areas(tb_area_1: ImageSegment, tb_area_2: ImageSegment, median_line_sep: float) -> bool:
    """
    Identify if two table areas are coherent
    :param tb_area_1: first table area
    :param tb_area_2: second table area
    :param median_line_sep: median line separation
    :return: boolean indicating if the two table areas are coherent
    """
    # Compute vertical difference
    v_diff = min(abs(tb_area_1.y2 - tb_area_2.y1), abs(tb_area_2.y2 - tb_area_1.y1))

    # If areas are not consecutive or with two much separation, not coherent
    if abs(tb_area_1.position - tb_area_2.position) != 1 or v_diff > 3 * median_line_sep:
        return False

    # Check whitespaces coherency
    if len(tb_area_1.whitespaces) >= len(tb_area_2.whitespaces):
        dict_ws_coherency = {
            idx_1: [ws_2 for ws_2 in tb_area_2.whitespaces if min(ws_1.x2, ws_2.x2) - max(ws_1.x1, ws_2.x1) > 0]
            for idx_1, ws_1 in enumerate(tb_area_1.whitespaces)
        }
    else:
        dict_ws_coherency = {
            idx_2: [ws_1 for ws_1 in tb_area_1.whitespaces if min(ws_1.x2, ws_2.x2) - max(ws_1.x1, ws_2.x1) > 0]
            for idx_2, ws_2 in enumerate(tb_area_2.whitespaces)
        }

    return np.mean([int(len(v) == 1) for v in dict_ws_coherency.values()]) >= 0.75


def table_segment_from_group(table_segment_group: List[ImageSegment]) -> ImageSegment:
    """
    Create table segment from group of corresponding ImageSegment objects
    :param table_segment_group: list of ImageSegment objects
    :return: ImageSegment corresponding to table
    """
    # Retrieve all elements
    elements = [el for seg in table_segment_group for el in seg.elements]
    whitespaces = [ws for seg in table_segment_group for ws in seg.whitespaces]

    # Create ImageSegment object
    table_segment = ImageSegment(x1=min([seg.x1 for seg in table_segment_group]) - 10,
                                 y1=min([seg.y1 for seg in table_segment_group]),
                                 x2=max([seg.x2 for seg in table_segment_group]) + 10,
                                 y2=max([seg.y2 for seg in table_segment_group]),
                                 elements=elements,
                                 whitespaces=whitespaces)
    
    return table_segment


def get_table_segments(segment: ImageSegment, char_length: float, median_line_sep: float) -> List[ImageSegment]:
    """
    Identify relevant table areas in segment
    :param segment: ImageSegment object
    :param char_length: average character length in image
    :param median_line_sep: median line separation
    :return: list of image segments corresponding to tables
    """
    # Get table areas
    table_areas = get_table_areas(segment=segment, char_length=char_length, median_line_sep=median_line_sep)

    if len(table_areas) == 0:
        return []

    # Create groups of table areas
    table_areas = sorted(table_areas, key=lambda tb: tb.position)
    seq = iter(table_areas)
    tb_areas_gps = [[next(seq)]]
    for tb_area in seq:
        prev_table = tb_areas_gps[-1][-1]
        if not coherent_table_areas(tb_area_1=prev_table, tb_area_2=tb_area, median_line_sep=median_line_sep):
            tb_areas_gps.append([])
        tb_areas_gps[-1].append(tb_area)
        
    # Create image segments corresponding to potential table
    table_segments = [table_segment_from_group(table_segment_group=gp) for gp in tb_areas_gps]

    return table_segments
