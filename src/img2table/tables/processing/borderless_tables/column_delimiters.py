# coding: utf-8

from queue import PriorityQueue
from typing import NamedTuple, List

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import LineGroup
from img2table.tables.processing.common import is_contained_cell


class Rectangle(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_cell(cls, cell: Cell) -> "Rectangle":
        return cls(x1=cell.x1, y1=cell.y1, x2=cell.x2, y2=cell.y2)

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def distance(self, other):
        return (self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2

    def overlaps(self, other):
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        return max(x_right - x_left, 0) * max(y_bottom - y_top, 0) > 0


def identify_trivial_delimiters(line_group: LineGroup, elements: List[Cell]) -> List[Cell]:
    """
    Get trivial whitespace delimiters spanning the entire line group height
    :param line_group: cluster of lines
    :param elements: Cell elements from image
    :return: list of trivial whitespace column delimiters
    """
    # Get all x values
    x_values = sorted(list(set([el.x1 for el in elements] + [el.x2 for el in elements])))

    # Identify trivial delimiters
    trivial_delimiters = list()
    for x_left, x_right in zip(x_values, x_values[1:]):
        # Get overlapping elements
        overlapping_elements = [el for el in elements if min(el.x2, x_right) - max(el.x1, x_left) > 0]

        # If no overlapping elements are found, create delimiter
        if len(overlapping_elements) == 0:
            delimiter = Cell(x1=x_left, x2=x_right, y1=line_group.y1, y2=line_group.y2)
            trivial_delimiters.append(delimiter)

    return trivial_delimiters


def find_whitespaces(line_group: LineGroup, elements: List[Cell]) -> List[Cell]:
    """
    Identify whitespaces in line group
    :param line_group: LineGroup object
    :param elements: elements from segment
    :return: list of whitespaces as Cell objects
    """
    # Create rectangle objects from inputs
    group_rect = Rectangle(x1=line_group.x1, y1=line_group.y1, x2=line_group.x2, y2=line_group.y2)
    obstacles = [Rectangle.from_cell(cell=element) for element in elements]

    # Initiate queue
    queue = PriorityQueue()
    queue.put([-group_rect.area, group_rect, obstacles])

    whitespaces = list()
    while not queue.qsize() == 0:
        q, r, obs = queue.get()
        if len(obs) == 0:
            # Update whitespaces and covered area
            whitespaces.append(r)

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
            if rect.area > group_rect.area / 1000:
                rect_obstacles = [o for o in obs if o.overlaps(rect)]
                queue.put([-rect.area, rect, rect_obstacles])

    return [space.cell for space in whitespaces]


def compute_vertical_delimiters(whitespaces: List[Cell], ref_height: int) -> List[Cell]:
    """
    Identify vertical column delimiters from computed whitespaces
    :param whitespaces: list of identified whitespace areas
    :param ref_height: reference height for vertical whitespace, represent line group height
    :return: list of identified vertical column delimiters
    """
    x_values = sorted(list(set([w.x1 for w in whitespaces] + [w.x2 for w in whitespaces])))

    v_delimiters = list()
    for x_left, x_right in zip(x_values, x_values[1:]):
        # Get overlapping whitespaces
        overlapping_ws = [ws for ws in whitespaces if min(ws.x2, x_right) - max(ws.x1, x_left) > 0]
        overlapping_ws = sorted(overlapping_ws, key=lambda ws: ws.y1 + ws.y2)

        if overlapping_ws:
            # Create groups with consecutive whitespaces
            seq = iter(overlapping_ws)
            ws_groups = [[next(seq)]]
            for ws in seq:
                if ws.y1 > ws_groups[-1][-1].y2:
                    ws_groups.append([])
                ws_groups[-1].append(ws)

            # If the group height is larger than 75% of the reference height, create a vertical delimiter
            for gp in [gp for gp in ws_groups if gp[-1].y2 - gp[0].y1 > ref_height * 0.75]:
                v_delimiter = Cell(x1=x_left, x2=x_right, y1=gp[0].y1, y2=gp[-1].y2)
                v_delimiters.append(v_delimiter)

    if len(v_delimiters) == 0:
        return v_delimiters

    # Merge consecutive delimiters that are adjacent and have the same height
    v_delimiters = sorted(v_delimiters, key=lambda v: v.x1 + v.x2)
    seq = iter(v_delimiters)
    del_groups = [[next(seq)]]
    for delim in seq:
        last_del = del_groups[-1][-1]
        if (delim.x1 != last_del.x2) or (delim.y1 != last_del.y1) or (delim.y2 != last_del.y2):
            del_groups.append([])
        del_groups[-1].append(delim)

    return [Cell(x1=min([el.x1 for el in del_group]),
                 y1=min([el.y1 for el in del_group]),
                 x2=max([el.x2 for el in del_group]),
                 y2=max([el.y2 for el in del_group]))
            for del_group in del_groups]


def filter_coherent_dels(v_delimiters: List[Cell], elements: List[Cell]) -> List[Cell]:
    """
    Filter pertinent delimiters and get most coherent ones
    :param v_delimiters: list of identified vertical delimiters
    :param elements: Cell elements from image
    :return: list of coherent vertical column delimiters
    """
    # Remove delimiters that are not pertinent, i.e it does not have elements to its left and its right
    pertinent_dels = list()
    for v_del in v_delimiters:
        # Get vertically overlapping elements
        v_overlap_els = [el for el in elements if min(v_del.y2, el.y2) - max(v_del.y1, el.y1) > 0]

        # Get elements to the left and right
        left_elements = [el for el in v_overlap_els if el.x2 <= v_del.x1]
        right_elements = [el for el in v_overlap_els if el.x1 >= v_del.x2]
        if len(left_elements) > 0 and len(right_elements) > 0:
            pertinent_dels.append(v_del)

    if len(pertinent_dels) == 0:
        return pertinent_dels

    # If two or more delimiters are adjacent, keep the tallest one
    clusters = list()
    for i in range(len(pertinent_dels)):
        for j in range(i, len(pertinent_dels)):
            intersect = {pertinent_dels[i].x1, pertinent_dels[i].x2}.intersection({pertinent_dels[j].x1, pertinent_dels[j].x2})
            # If delimiters are adjacent, find matching clusters
            if len(intersect) > 0:
                matching_clusters = [idx for idx, cl in enumerate(clusters) if {i, j}.intersection(cl)]
                if matching_clusters:
                    remaining_clusters = [cl for idx, cl in enumerate(clusters) if idx not in matching_clusters]
                    new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(clusters) if idx in matching_clusters])
                    clusters = remaining_clusters + [new_cluster]
                else:
                    clusters.append({i, j})

    # Get coherent delimiters
    coherent_dels = list()
    max_height = max([delim.height for delim in pertinent_dels])
    for cl in clusters:
        cl_dels = [pertinent_dels[idx] for idx in cl]
        max_height_delims = [delim for delim in cl_dels if delim.height == max_height]
        if max_height_delims:
            coherent_dels += max_height_delims
            continue
        coherent_dels.append(sorted(cl_dels, key=lambda d: (d.height, d.area)).pop())

    # Normalize heights and set minimal width
    y_min = max([delim.y1 for delim in coherent_dels])
    y_max = min([delim.y2 for delim in coherent_dels])

    return [Cell(x1=delim.x1, y1=y_min, x2=delim.x2, y2=y_max) for delim in coherent_dels
            if delim.width >= 5]


def get_whitespace_column_delimiters(line_group: LineGroup, segment_elements: List[Cell]) -> List[Cell]:
    """
    For a line group, identify coherent vertical delimiters
    :param line_group: group of lines as LineGroup object
    :param segment_elements: list of Cell elements from the corresponding image segment
    :return: list of vertical column delimiters corresponding to line group
    """
    # Get elements that correspond to line group
    lg_elements = [element for element in segment_elements
                   if is_contained_cell(inner_cell=element, outer_cell=line_group)]

    # Identify trivial delimiters
    trivial_delimiters = identify_trivial_delimiters(line_group=line_group,
                                                     elements=lg_elements)

    # Identify whitespaces
    whitespaces = find_whitespaces(line_group=line_group,
                                   elements=lg_elements+trivial_delimiters)

    # Identify vertical delimiters
    vertical_dels = compute_vertical_delimiters(whitespaces=whitespaces,
                                                ref_height=line_group.height)

    # Get coherent delimiters
    coherent_dels = filter_coherent_dels(v_delimiters=trivial_delimiters + vertical_dels,
                                         elements=lg_elements)

    return coherent_dels
