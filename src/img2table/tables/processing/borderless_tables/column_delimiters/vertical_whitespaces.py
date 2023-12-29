# coding: utf-8

from dataclasses import dataclass, field
from typing import List, Tuple

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import TableSegment
from img2table.tables.processing.borderless_tables.whitespaces import adjacent_whitespaces


@dataclass
class MatchingWS:
    ws: Cell
    position: int
    top: bool
    bottom: bool


@dataclass
class VertWS:
    x1: int
    x2: int
    y1: int
    y2: int
    whitespaces: List[Cell] = field(default_factory=lambda: [])

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)


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
    table_areas = sorted(table_segment.table_areas, key=lambda x: x.position)

    # Identify all whitespaces x values
    x_ws = sorted(set([ws.x1 for ws in table_segment.whitespaces] + [ws.x2 for ws in table_segment.whitespaces]))

    # Get vertical whitespaces
    vertical_ws = list()
    for x_left, x_right in zip(x_ws, x_ws[1:]):
        rng_ws = list()
        for id_area, tb_area in enumerate(table_areas):
            # Get matching whitespaces
            matching_ws = sorted([ws for ws in tb_area.whitespaces if min(x_right, ws.x2) - max(x_left, ws.x1) > 0],
                                 key=lambda ws: ws.y1)

            if matching_ws:
                for ws in matching_ws:
                    m_ws = MatchingWS(ws=ws,
                                      position=id_area,
                                      top=ws.y1 == tb_area.y1,
                                      bottom=ws.y2 == tb_area.y2)
                    rng_ws.append(m_ws)

        if rng_ws:
            # Create cluster of coherent ws
            seq = iter(rng_ws)
            ws_clusters = [[next(seq)]]
            for m_ws in seq:
                prev_ws = ws_clusters[-1][-1]

                # If consecutive ws do not match, create a new cluster
                if m_ws.position - prev_ws.position > 1 or not (prev_ws.bottom and m_ws.top):
                    ws_clusters.append([])
                ws_clusters[-1].append(m_ws)

            for cl in ws_clusters:
                # Compute vertical boundaries
                y1 = table_areas[cl[0].position - 1].y2 if cl[0].top and cl[0].position > 0 else cl[0].ws.y1
                y2 = table_areas[cl[-1].position + 1].y1 if cl[-1].bottom and cl[-1].position < len(table_areas) - 1 else cl[-1].ws.y2

                vert_ws = VertWS(x1=x_left, x2=x_right, y1=y1, y2=y2,
                                 whitespaces=[m_ws.ws for m_ws in cl])
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
