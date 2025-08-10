import random
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Union

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.model import ImageSegment
from img2table.tables.processing.borderless_tables.whitespaces import get_whitespaces


@dataclass
class Rectangle:
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_cell(cls, cell: Cell) -> "Rectangle":
        return cls(x1=cell.x1, y1=cell.y1, x2=cell.x2, y2=cell.y2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def distance(self, other: "Rectangle") -> float:
        return (self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2

    def overlaps(self, other: "Rectangle") -> bool:
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        return max(x_right - x_left, 0) * max(y_bottom - y_top, 0) > 0


def identify_remaining_segments(searched_rectangle: Rectangle,
                                existing_segments: list[Union[Cell, ImageSegment]]) -> list[Cell]:
    """
    Identify remaining segments in searched rectangle
    :param searched_rectangle: rectangle corresponding to area of research
    :param existing_segments: list of existing image segments
    :return: list of whitespaces as Cell objects
    """
    # Create rectangle objects from inputs
    obstacles = [Rectangle.from_cell(cell=element) for element in existing_segments]

    # Initiate queue
    queue = PriorityQueue()
    queue.put([-searched_rectangle.area, searched_rectangle, obstacles])

    segments = []
    while queue.qsize() != 0:
        q, r, obs = queue.get()
        if len(obs) == 0:
            # Update segments
            segments.append(r)

            # Update elements in queue
            for element in queue.queue:
                if element[1].overlaps(r):
                    element[2] += [r]

            continue

        # Get most pertinent obstacle
        pivot = sorted(obs, key=lambda o: o.distance(r))[0]

        # Create new rectangles
        rects = [Rectangle(x1=pivot.x2, y1=r.y1, x2=r.x2, y2=r.y2),
                 Rectangle(x1=r.x1, y1=r.y1, x2=pivot.x1, y2=r.y2),
                 Rectangle(x1=r.x1, y1=pivot.y2, x2=r.x2, y2=r.y2),
                 Rectangle(x1=r.x1, y1=r.y1, x2=r.x2, y2=pivot.y1)]

        for rect in rects:
            if rect.area > searched_rectangle.area / 100:
                rect_obstacles = [o for o in obs if o.overlaps(rect)]
                queue.put([-rect.area + random.uniform(0, 1), rect, rect_obstacles])

    return [seg.cell for seg in segments]


def get_vertical_ws(image_segment: ImageSegment, char_length: float, lines: list[Line]) -> list[Cell]:
    """
    Identify vertical whitespaces that can correspond to column delimiters in document
    :param image_segment: segment corresponding to the image
    :param char_length: average character length
    :param lines: list of lines identified in image
    :return: list of vertical whitespaces that can correspond to column delimiters in document
    """
    # Identify vertical whitespaces in segment that represent at least half of the image segment
    v_ws = get_whitespaces(segment=image_segment, vertical=True, pct=0.5)
    v_ws = [ws for ws in v_ws if ws.width >= char_length or ws.x1 == image_segment.x1 or ws.x2 == image_segment.x2]

    if len(v_ws) == 0:
        return []

    # Cut whitespaces with horizontal lines
    line_ws = []
    h_lines = [line for line in lines if line.horizontal]
    for ws in v_ws:
        # Get crossing h_lines
        crossing_h_lines = sorted([line for line in h_lines if ws.y1 < line.y1 < ws.y2
                                   and min(ws.x2, line.x2) - max(ws.x1, line.x1) >= 0.5 * ws.width],
                                  key=lambda line: line.y1)
        if len(crossing_h_lines) > 0:
            # Get y values from whitespace and crossing lines
            y_values = sorted([ws.y1, ws.y2]
                              + [line.y1 - line.thickness for line in crossing_h_lines]
                              + [line.y1 + line.thickness for line in crossing_h_lines])

            # Create new sub whitespaces that are between two horizontal lines
            for y_top, y_bottom in [y_values[idx:idx + 2] for idx in range(0, len(y_values), 2)]:
                if y_bottom - y_top >= 0.5 * image_segment.height:
                    new_ws = Cell(x1=ws.x1, y1=y_top, x2=ws.x2, y2=y_bottom)
                    line_ws.append(new_ws)
        else:
            line_ws.append(ws)

    if len(line_ws) == 0:
        return []

    # Create groups of adjacent whitespaces
    line_ws = sorted(line_ws, key=lambda ws: ws.x1 + ws.x2)
    seq = iter(line_ws)

    line_ws_groups = [[next(seq)]]
    for ws in seq:
        prev_ws = line_ws_groups[-1][-1]

        # Get area delimited by the two whitespaces
        x1_area, x2_area = min(prev_ws.x2, ws.x1), max(prev_ws.x2, ws.x1)
        y1_area, y2_area = max(prev_ws.y1, ws.y1), min(prev_ws.y2, ws.y2)
        area = Cell(x1=x1_area, y1=y1_area, x2=x2_area, y2=y2_area)

        # Get separating elements
        separating_elements = [el for el in image_segment.elements if el.x1 >= area.x1 and el.x2 <= area.x2
                               and el.y1 >= area.y1 and el.y2 <= area.y2]

        if len(separating_elements) > 0:
            line_ws_groups.append([])
        line_ws_groups[-1].append(ws)

    # Keep only the tallest whitespace in each group
    return [sorted([ws for ws in cl if ws.height == max([w.height for w in cl])], key=lambda w: w.area).pop()
            for cl in line_ws_groups]


def is_column_section(ws_group: list[Cell]) -> bool:
    """
    Identify if the whitespace group can correspond to columns
    :param ws_group: group of whitespaces
    :return: boolean indicating if the whitespace group can correspond to columns
    """
    # Check number of potential columns
    if not 3 <= len(ws_group) <= 4:
        return False

    # Check if column widths are consistent within the group
    ws_group = sorted(ws_group, key=lambda ws: ws.x1 + ws.x2)
    col_widths = [r_ws.x1 - l_ws.x2 for l_ws, r_ws in zip(ws_group, ws_group[1:])]

    return max(col_widths) / min(col_widths) <= 1.25


def top_matches(col_1: Cell, col_2: Cell) -> bool:
    """
    Identify if the top ends of columns are closely matching
    :param col_1: first column as cell
    :param col_2: second column as cell
    :return: boolean indicating if the top ends of columns are closely matching
    """
    return abs(col_1.y1 - col_2.y1) / max(col_1.height, col_2.height) <= 0.05


def bottom_matches(col_1: Cell, col_2: Cell) -> bool:
    """
    Identify if the bottom ends of columns are closely matching
    :param col_1: first column as cell
    :param col_2: second column as cell
    :return: boolean indicating if the bottom ends of columns are closely matching
    """
    return abs(col_1.y2 - col_2.y2) / max(col_1.height, col_2.height) <= 0.05


def identify_column_groups(image_segment: ImageSegment, vertical_ws: list[Cell]) -> list[list[Cell]]:
    """
    Identify groups of whitespaces that correspond to document columns
    :param image_segment: segment corresponding to the image
    :param vertical_ws: list of vertical whitespaces that can correspond to column delimiters in document
    :return: groups of whitespaces that correspond to document columns
    """
    # Identify whitespaces in the middle of the image as well as on edges
    middle_ws = [ws for ws in vertical_ws if
                 len({ws.x1, ws.x2}.intersection({image_segment.x1, image_segment.x2})) == 0]
    edge_ws = [ws for ws in vertical_ws if len({ws.x1, ws.x2}.intersection({image_segment.x1, image_segment.x2})) > 0]

    # Create groups of columns based on top/bottom alignment
    top_col_groups = [cl + edge_ws for cl in cluster_items(items=middle_ws, clustering_func=top_matches)]
    bottom_col_groups = [cl + edge_ws for cl in cluster_items(items=middle_ws, clustering_func=bottom_matches)]

    # Identify groups that correspond to columns
    col_groups = sorted([gp for gp in top_col_groups + bottom_col_groups if is_column_section(ws_group=gp)],
                        key=len,
                        reverse=True)

    # Get groups that contain all relevant whitespaces
    filtered_col_groups = []
    for col_gp in col_groups:
        y_min, y_max = min([ws.y1 for ws in col_gp]), max([ws.y2 for ws in col_gp])
        matching_ws = [ws for ws in vertical_ws if min(ws.y2, y_max) - max(ws.y1, y_min) > 0.2 * ws.height
                       and len({ws.x1, ws.x2}.intersection({image_segment.x1, image_segment.x2})) == 0]
        if len(set(matching_ws).difference(set(col_gp))) == 0:
            filtered_col_groups.append(col_gp)

    if len(filtered_col_groups) == 0:
        return []

    # Deduplicate column groups
    seq = iter(filtered_col_groups)
    dedup_col_groups = [next(seq)]
    for col_gp in seq:
        if not any(set(col_gp).intersection(set(gp)) == set(col_gp) for gp in dedup_col_groups):
            dedup_col_groups.append(col_gp)

    return dedup_col_groups


def get_column_group_segments(col_group: list[Cell]) -> list[ImageSegment]:
    """
    Identify image segments from the column group
    :param col_group: group of whitespaces that correspond to document columns
    :return: list of image segments defined by the column group
    """
    # Compute segments delimited by columns
    col_group = sorted(col_group, key=lambda ws: ws.x1 + ws.x2)
    col_segments = []

    for left_ws, right_ws in zip(col_group, col_group[1:]):
        y1_segment, y2_segment = max(left_ws.y1, right_ws.y1), min(left_ws.y2, right_ws.y2)
        x1_segment, x2_segment = round((left_ws.x1 + left_ws.x2) / 2), round((right_ws.x1 + right_ws.x2) / 2)
        segment = ImageSegment(x1=x1_segment, y1=y1_segment, x2=x2_segment, y2=y2_segment)
        col_segments.append(segment)

    # Create rectangle defined by segments and identify remaining segments in area
    cols_rectangle = Rectangle(x1=min([seg.x1 for seg in col_segments]),
                               y1=min([seg.y1 for seg in col_segments]),
                               x2=max([seg.x2 for seg in col_segments]),
                               y2=max([seg.y2 for seg in col_segments]))
    remaining_segments = [ImageSegment(x1=area.x1, y1=area.y1, x2=area.x2, y2=area.y2)
                          for area in identify_remaining_segments(searched_rectangle=cols_rectangle,
                                                                  existing_segments=col_segments)
                          ]

    return col_segments + remaining_segments


def get_segments_from_columns(image_segment: ImageSegment, column_groups: list[list[Cell]]) -> list[ImageSegment]:
    """
    Identify all segments in image from columns
    :param image_segment: segment corresponding to the image
    :param column_groups: groups of whitespaces that correspond to document columns
    :return: list of segments in image from columns
    """
    # Identify image segments from column groups
    col_group_segments = [seg for col_gp in column_groups
                          for seg in get_column_group_segments(col_group=col_gp)]

    # Identify segments outside of columns
    top_segment = ImageSegment(x1=image_segment.x1,
                               y1=image_segment.y1,
                               x2=image_segment.x2,
                               y2=min([seg.y1 for seg in col_group_segments]))
    bottom_segment = ImageSegment(x1=image_segment.x1,
                                  y1=max([seg.y2 for seg in col_group_segments]),
                                  x2=image_segment.x2,
                                  y2=image_segment.y2)
    left_segment = ImageSegment(x1=image_segment.x1,
                                y1=min([seg.y1 for seg in col_group_segments]),
                                x2=min([seg.x1 for seg in col_group_segments]),
                                y2=max([seg.y2 for seg in col_group_segments]))
    right_segment = ImageSegment(x1=max([seg.x2 for seg in col_group_segments]),
                                 y1=min([seg.y1 for seg in col_group_segments]),
                                 x2=image_segment.x2,
                                 y2=max([seg.y2 for seg in col_group_segments]))

    # Create image segments and identify missing segments
    img_segments = [*col_group_segments, top_segment, bottom_segment, left_segment, right_segment]
    missing_segments = [ImageSegment(x1=area.x1, y1=area.y1, x2=area.x2, y2=area.y2)
                        for area in identify_remaining_segments(searched_rectangle=Rectangle.from_cell(image_segment),
                                                                existing_segments=img_segments)
                        ]

    return img_segments + missing_segments


def segment_image_columns(image_segment: ImageSegment, char_length: float, lines: list[Line]) -> list[ImageSegment]:
    """
    Create image segments by identifying columns
    :param image_segment: segment corresponding to the image
    :param char_length: average character length
    :param lines: list of lines identified in image
    :return: list of segments corresponding to image
    """
    # Identify vertical whitespaces that can correspond to column delimiters in document
    vertical_ws = get_vertical_ws(image_segment=image_segment,
                                  char_length=char_length,
                                  lines=lines)

    # Identify column groups
    column_groups = identify_column_groups(image_segment=image_segment,
                                           vertical_ws=vertical_ws)

    if len(column_groups) == 0:
        return [image_segment]

    # Identify all segments in image from columns
    col_segments = get_segments_from_columns(image_segment=image_segment,
                                             column_groups=column_groups)

    # Populate elements in groups
    final_segments = []
    for segment in col_segments:
        segment_elements = [el for el in image_segment.elements if el.x1 >= segment.x1 and el.x2 <= segment.x2
                            and el.y1 >= segment.y1 and el.y2 <= segment.y2]
        if segment_elements:
            segment.set_elements(elements=segment_elements)
            final_segments.append(segment)

    return final_segments
