
import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import ImageSegment, TableSegment, Whitespace
from img2table.tables.processing.borderless_tables.whitespaces import get_whitespaces, \
    get_relevant_vertical_whitespaces
from img2table.tables.processing.common import is_contained_cell


def get_table_areas(segment: ImageSegment, char_length: float, median_line_sep: float) -> list[ImageSegment]:
    """
    Identify relevant table areas in segment
    :param segment: ImageSegment object
    :param char_length: average character length in image
    :param median_line_sep: median line separation
    :return: list of table areas of segment
    """
    # Identify horizontal whitespaces in segment that represent at least half of median line separation
    h_ws = get_whitespaces(segment=segment, vertical=False, pct=1, min_width=0.5 * median_line_sep)
    h_ws = sorted(h_ws, key=lambda ws: ws.y1)

    # Handle case where no whitespaces have been found by creating "fake" ws at the top or bottom
    if len(h_ws) == 0:
        h_ws = [Whitespace(cells=[Cell(x1=min([el.x1 for el in segment.elements]),
                                       x2=max([el.x2 for el in segment.elements]),
                                       y1=segment.y1,
                                       y2=segment.y1)]),
                Whitespace(cells=[Cell(x1=min([el.x1 for el in segment.elements]),
                                       x2=max([el.x2 for el in segment.elements]),
                                       y1=segment.y2,
                                       y2=segment.y2)])
                ]

    # Create whitespaces at the top or the bottom if they are missing
    if h_ws[0].y1 > segment.y1:
        up_ws = Whitespace(cells=[Cell(x1=min([ws.x1 for ws in h_ws]),
                                       x2=max([ws.x2 for ws in h_ws]),
                                       y1=segment.y1,
                                       y2=min([el.y1 for el in segment.elements]))])
        h_ws.insert(0, up_ws)

    if h_ws[-1].y2 < segment.y2:
        down_ws = Whitespace(cells=[Cell(x1=min([ws.x1 for ws in h_ws]),
                                         x2=max([ws.x2 for ws in h_ws]),
                                         y1=segment.y2,
                                         y2=max([el.y2 for el in segment.elements]))])
        h_ws.append(down_ws)

    # Check in areas between horizontal whitespaces in order to identify if they can correspond to tables
    table_areas = []

    for idx, (up, down) in enumerate(zip(h_ws, h_ws[1:])):
        # Get the delimited area
        delimited_area = Cell(x1=max(min(up.x1, down.x1) - int(char_length), 0),
                              y1=up.y2,
                              x2=min(max(up.x2, down.x2) + int(char_length), segment.x2),
                              y2=down.y1)

        # Identify corresponding elements and create a corresponding segment
        area_elements = [el for el in segment.elements if el.x1 >= delimited_area.x1 and el.x2 <= delimited_area.x2
                         and el.y1 >= delimited_area.y1 and el.y2 <= delimited_area.y2]
        seg_area = ImageSegment(x1=delimited_area.x1,
                                x2=delimited_area.x2,
                                y1=delimited_area.y1,
                                y2=delimited_area.y2,
                                elements=area_elements,
                                position=idx + 1)

        if area_elements:
            # Identify vertical whitespaces in the area
            v_ws = get_relevant_vertical_whitespaces(segment=seg_area,
                                                     char_length=char_length,
                                                     median_line_sep=median_line_sep,
                                                     pct=0.66)

            # Identify number of whitespaces that are not on borders
            middle_ws = [ws for ws in v_ws if ws.x1 != seg_area.x1 and ws.x2 != seg_area.x2]

            # If there can be at least 3 columns in area, it is a possible table area
            if len(middle_ws) >= 1:
                # Add edges whitespaces
                left_ws = Whitespace(cells=[Cell(x1=seg_area.x1,
                                                 y1=seg_area.y1,
                                                 x2=min([el.x1 for el in seg_area.elements]),
                                                 y2=seg_area.y2)])
                right_ws = Whitespace(cells=[Cell(x1=max([el.x2 for el in seg_area.elements]),
                                                  y1=seg_area.y1,
                                                  x2=seg_area.x2,
                                                  y2=seg_area.y2)])
                v_ws = [ws for ws in v_ws
                        if not is_contained_cell(inner_cell=ws, outer_cell=left_ws, percentage=0.1)
                        and not is_contained_cell(inner_cell=ws, outer_cell=right_ws, percentage=0.1)
                        and ((len({ws.y1, ws.y2}.intersection({seg_area.y1, seg_area.y2})) > 0)
                             or (ws.height >= 0.66 * max([w.height for w in middle_ws])))]

                seg_area.set_whitespaces(whitespaces=sorted([*v_ws, left_ws, right_ws], key=lambda ws: ws.x1 + ws.x2))
                table_areas.append(seg_area)

    return table_areas


def merge_consecutive_ws(whitespaces: list[Whitespace]) -> list[Cell]:
    """
    Merge consecutive whitespaces
    :param whitespaces: list of original whitespaces
    :return: list of merged whitespaces
    """
    whitespaces = sorted(whitespaces, key=lambda ws: ws.x1 + ws.x2)

    seq = iter(whitespaces)
    ws_groups = [[next(seq)]]
    for ws in seq:
        if ws.x1 > ws_groups[-1][-1].x2:
            ws_groups.append([])
        ws_groups[-1].append(ws)

    return [Cell(x1=ws_gp[0].x1, x2=ws_gp[-1].x2,
                 y1=min([ws.y1 for ws in ws_gp]),
                 y2=max([ws.y2 for ws in ws_gp]))
            for ws_gp in ws_groups]


def coherent_table_areas(tb_area_1: ImageSegment, tb_area_2: ImageSegment, char_length: float, median_line_sep: float) -> bool:
    """
    Identify if two table areas are coherent
    :param tb_area_1: first table area
    :param tb_area_2: second table area
    :param char_length: average character length in image
    :param median_line_sep: median line separation
    :return: boolean indicating if the two table areas are coherent
    """
    # Compute vertical difference
    v_diff = max(tb_area_1.y1, tb_area_2.y1) - min(tb_area_1.y2, tb_area_2.y2)

    # If areas are not consecutive or with too much separation, not coherent
    if abs(tb_area_1.position - tb_area_2.position) != 1 or v_diff > 2.5 * median_line_sep:
        return False

    # Get relevant whitespaces
    if tb_area_1.position < tb_area_2.position:
        ws_tb_1 = merge_consecutive_ws([ws for ws in tb_area_1.whitespaces if ws.y2 == tb_area_1.y2])
        ws_tb_2 = merge_consecutive_ws([ws for ws in tb_area_2.whitespaces if ws.y1 == tb_area_2.y1])
    else:
        ws_tb_1 = merge_consecutive_ws([ws for ws in tb_area_1.whitespaces if ws.y1 == tb_area_1.y1])
        ws_tb_2 = merge_consecutive_ws([ws for ws in tb_area_2.whitespaces if ws.y2 == tb_area_2.y2])

    # Check whitespaces coherency on "middle" whitespaces
    if len(ws_tb_1) >= len(ws_tb_2):
        dict_ws_coherency = {
            idx_1: [ws_2 for ws_2 in ws_tb_2
                    if min(ws_1.x2, ws_2.x2) - max(ws_1.x1, ws_2.x1) >= 0.5 * char_length]
            for idx_1, ws_1 in enumerate(ws_tb_1[1:-1])
        }
    else:
        dict_ws_coherency = {
            idx_2: [ws_1 for ws_1 in ws_tb_1
                    if min(ws_1.x2, ws_2.x2) - max(ws_1.x1, ws_2.x1) >= 0.5 * char_length]
            for idx_2, ws_2 in enumerate(ws_tb_2[1:-1])
        }

    # Compute threshold for coherency
    if min(len(ws_tb_1), len(ws_tb_2)) < 4:
        threshold = 1
    elif v_diff < median_line_sep:
        threshold = 0.66
    else:
        threshold = 0.8

    return np.mean([int(len(v) == 1) for v in dict_ws_coherency.values()]) >= threshold


def table_segment_from_group(table_segment_group: list[ImageSegment]) -> ImageSegment:
    """
    Create table segment from group of corresponding ImageSegment objects
    :param table_segment_group: list of ImageSegment objects
    :return: ImageSegment corresponding to table
    """
    # Retrieve all elements
    elements = [el for seg in table_segment_group for el in seg.elements]
    whitespaces = [ws for seg in table_segment_group for ws in seg.whitespaces]

    # Create ImageSegment object
    return ImageSegment(x1=min([seg.x1 for seg in table_segment_group]),
                        y1=min([seg.y1 for seg in table_segment_group]),
                        x2=max([seg.x2 for seg in table_segment_group]),
                        y2=max([seg.y2 for seg in table_segment_group]),
                        elements=elements,
                        whitespaces=whitespaces)


def get_table_segments(segment: ImageSegment, char_length: float, median_line_sep: float) -> list[TableSegment]:
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
        if not coherent_table_areas(tb_area_1=prev_table,
                                    tb_area_2=tb_area,
                                    char_length=char_length,
                                    median_line_sep=median_line_sep):
            tb_areas_gps.append([])
        tb_areas_gps[-1].append(tb_area)

    # Create image segments corresponding to potential table
    return [TableSegment(table_areas=tb_area_gp) for tb_area_gp in tb_areas_gps
            if max([len(tb_area.whitespaces) for tb_area in tb_area_gp]) > 3]
