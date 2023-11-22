# coding: utf-8

from dataclasses import dataclass, field
from typing import List, Tuple

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import TableSegment
from img2table.tables.processing.borderless_tables.whitespaces import adjacent_whitespaces


@dataclass
class VertWS:
    x1: int
    x2: int
    whitespaces: List[Cell] = field(default_factory=lambda: [])
    positions: List[int] = field(default_factory=lambda: [])

    @property
    def y1(self):
        return min([ws.y1 for ws in self.whitespaces]) if self.whitespaces else 0

    @property
    def y2(self):
        return max([ws.y2 for ws in self.whitespaces]) if self.whitespaces else 0

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    @property
    def continuous(self):
        if self.positions:
            positions = sorted(self.positions)
            return all([p2 - p1 <= 1 for p1, p2 in zip(positions, positions[1:])])
        return False

    def add_ws(self, whitespaces: List[Cell]):
        self.whitespaces += whitespaces

    def add_position(self, position: int):
        self.positions.append(position)


def deduplicate_whitespaces(vertical_whitespaces: List[VertWS], elements: List[Cell]) -> List[VertWS]:
    """
    Deduplicate adjacent vertical whitespaces
    :param vertical_whitespaces: list of VertWS objects
    :param elements: list of elements in segment
    :return: deduplicated vertical whitespaces
    """
    # Identify maximum height of whitespaces
    max_ws_height = max([ws.height for ws in vertical_whitespaces])

    # Create clusters of adjacent whitespaces
    ws_clusters = cluster_items(items=vertical_whitespaces,
                                clustering_func=adjacent_whitespaces)

    # For each group, get the tallest whitespace
    dedup_ws = list()
    for cl in ws_clusters:
        # Get x center of cluster
        x_center = min([ws.x1 for ws in cl]) + max([ws.x2 for ws in cl])

        # Filter on tallest delimiters
        max_cl_height = max([ws.height for ws in cl])
        tallest_ws = [ws for ws in cl if ws.height == max_cl_height]

        if max_cl_height == max_ws_height:
            dedup_ws += tallest_ws
        else:
            # Add whitespace closest to the center of the group
            closest_ws = sorted(tallest_ws, key=lambda ws: abs(ws.x1 + ws.x2 - x_center)).pop(0)
            dedup_ws.append(closest_ws)

    # Finally remove consecutive whitespaces that do not have elements between them
    dedup_ws = sorted(dedup_ws, key=lambda ws: ws.x1 + ws.x2)
    ws_to_del = list()
    for ws_left, ws_right in zip(dedup_ws, dedup_ws[1:]):
        # Get common area
        common_area = Cell(x1=ws_left.x2,
                           y1=max(ws_left.y1, ws_right.y1),
                           x2=ws_right.x1,
                           y2=min(ws_left.y2, ws_right.y2))

        # Identify matching elements
        matching_elements = [el for el in elements if el.x1 >= common_area.x1 and el.x2 <= common_area.x2
                             and el.y1 >= common_area.y1 and el.y2 <= common_area.y2]

        if len(matching_elements) == 0:
            # Add smallest element to deleted ws
            ws_to_del.append(ws_left if ws_left.height < ws_right.height else ws_right)

    return [ws for ws in dedup_ws if ws not in ws_to_del]


def get_vertical_whitespaces(table_segment: TableSegment) -> Tuple[List[Cell], List[Cell]]:
    """
    Identify vertical whitespaces as well as unused whitespaces in table segment
    :param table_segment: TableSegment object
    :return: tuple containing list of vertical whitespaces and list of unused whitespaces
    """
    # Identify all whitespaces x values
    x_ws = sorted(set([ws.x1 for ws in table_segment.whitespaces] + [ws.x2 for ws in table_segment.whitespaces]))

    # Get vertical whitespaces
    vertical_ws = list()
    for x_left, x_right in zip(x_ws, x_ws[1:]):
        # Create a whitespace object
        vert_ws = VertWS(x1=x_left, x2=x_right)

        for tb_area in table_segment.table_areas:
            # Get matching whitespaces
            matching_ws = [ws for ws in tb_area.whitespaces if min(vert_ws.x2, ws.x2) - max(vert_ws.x1, ws.x1) > 0]

            if matching_ws:
                vert_ws.add_position(tb_area.position)
                vert_ws.add_ws(matching_ws)

        # If it is composed of continuous whitespaces, use them
        if vert_ws.continuous:
            vertical_ws.append(vert_ws)

    # Filter whitespaces by height
    max_height = max([ws.height for ws in vertical_ws])
    vertical_ws = [ws for ws in vertical_ws if ws.height >= 0.66 * max_height]

    # Identify segment whitespaces that are unused
    unused_ws = [ws for ws in table_segment.whitespaces
                 if ws not in [ws for v_ws in vertical_ws for ws in v_ws.whitespaces]]

    # Deduplicate adjacent vertical delimiters
    vertical_ws = deduplicate_whitespaces(vertical_whitespaces=vertical_ws,
                                          elements=table_segment.elements)

    return [ws.cell for ws in vertical_ws], unused_ws
