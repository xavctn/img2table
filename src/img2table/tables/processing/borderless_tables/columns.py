# coding: utf-8

import copy
from dataclasses import dataclass
from typing import Optional, List

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import TableSegment, DelimiterGroup


@dataclass
class VerticalWS:
    ws: Cell
    position: int
    top: bool
    bottom: bool
    used: bool = False

    @property
    def x1(self) -> int:
        return self.ws.x1

    @property
    def y1(self) -> int:
        return self.ws.y1

    @property
    def x2(self) -> int:
        return self.ws.x2

    @property
    def y2(self) -> int:
        return self.ws.y2

    @property
    def width(self) -> int:
        return self.ws.x2 - self.ws.x1


@dataclass
class WSGroup:
    x1: int
    y1: int
    x2: int
    y2: int
    top: bool
    bottom: bool
    top_position: int
    bottom_position: int

    @classmethod
    def from_ws(cls, v_ws: VerticalWS) -> "WSGroup":
        return cls(x1=v_ws.x1, y1=v_ws.y1, x2=v_ws.x2, y2=v_ws.y2, top=v_ws.top, bottom=v_ws.bottom,
                   top_position=v_ws.position, bottom_position=v_ws.position)

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def corresponds(self, v_ws: VerticalWS, char_length: float) -> bool:
        if self.bottom_position is None:
            return True
        elif v_ws.position != self.bottom_position + 1:
            return False
        elif not self.bottom or not v_ws.top:
            return False

        # Condition on position
        return min(self.x2, v_ws.x2) - max(self.x1, v_ws.x1) >= 0.5 * char_length

    def add(self, v_ws: VerticalWS):
        self.x1 = max(self.x1, v_ws.x1)
        self.y1 = min(self.y1, v_ws.y1)
        self.x2 = min(self.x2, v_ws.x2)
        self.y2 = max(self.y2, v_ws.y2)
        self.top_position = min(self.top_position, v_ws.position)
        self.bottom_position = max(self.bottom_position, v_ws.position)

        if v_ws.position == self.top_position:
            self.top = v_ws.top

        if v_ws.position == self.bottom_position:
            self.bottom = v_ws.bottom


def get_columns_delimiters(table_segment: TableSegment, char_length: float) -> List[Cell]:
    """
    Identify column delimiters in table segment
    :param table_segment: TableSegment object
    :param char_length: average character length
    :return: list of column delimiters
    """
    # Get whitespaces
    table_areas = sorted(table_segment.table_areas, key=lambda x: x.position)

    # Create groups of relevant vertical whitespaces
    ws_groups = list()
    for id_area, tb_area in enumerate(table_areas):
        new_groups = list()
        whitespaces = [VerticalWS(ws=ws,
                                  top=ws.y1 == tb_area.y1,
                                  bottom=ws.y2 == tb_area.y2,
                                  position=id_area)
                       for ws in tb_area.whitespaces]

        for gp in ws_groups:
            # Get matching whitespaces
            matching_ws = [v_ws for v_ws in whitespaces if gp.corresponds(v_ws=v_ws, char_length=char_length)]

            if matching_ws:
                for v_ws in matching_ws:
                    # Update whitespace
                    setattr(v_ws, "used", True)

                    # Create new group
                    new_gp = copy.deepcopy(gp)
                    new_gp.add(v_ws)
                    new_groups.append(new_gp)
            else:
                new_groups.append(gp)

        # Create groups corresponding to unused whitespaces
        new_groups += [WSGroup.from_ws(v_ws=v_ws) for v_ws in whitespaces if not v_ws.used]

        # Replace existing groups by new groups
        ws_groups = new_groups

    # Recompute boundaries of whitespaces groups (up to previous/next area)
    dict_bounds = {k: {"y_min": table_areas[k].y1, "y_max": table_areas[k].y2}
                   for k in range(len(table_areas))}
    ws_cells = [Cell(x1=gp.x1,
                     y1=dict_bounds.get(gp.top_position - 1, {}).get("y_max") or gp.y1 if gp.top else gp.y1,
                     x2=gp.x2,
                     y2=dict_bounds.get(gp.bottom_position + 1, {}).get("y_min") or gp.y2 if gp.bottom else gp.y2)
                for gp in ws_groups]

    # Keep only whitespaces that represent at least 66% of the maximum height
    max_height = max(map(lambda ws: ws.height, ws_cells))
    ws_cells = [ws for ws in ws_cells if ws.height >= 0.66 * max_height]

    return ws_cells


def get_relevant_height(whitespaces: List[Cell], elements: List[Cell], char_length: float) -> List[Cell]:
    """
    Get relevant whitespaces height relative to image elements
    :param whitespaces: list of whitespaces defining column delimiters
    :param elements: list of image elements
    :param char_length: average character length
    :return: list of resized column delimiters
    """
    # Group elements in rows
    seq = iter(sorted(elements, key=lambda el: (el.y1, el.y2)))
    rows = [[next(seq)]]
    for el in seq:
        y2_row = max([el.y2 for el in rows[-1]])
        if el.y1 >= y2_row:
            rows.append([])
        rows[-1].append(el)

    # Identify top and bottom values for vertical whitespaces
    y_top, y_bottom, = max([ws.y2 for ws in whitespaces]), min([ws.y1 for ws in whitespaces])
    for row in rows:
        x1_row, x2_row = min([el.x1 for el in row]), max([el.x2 for el in row])
        y1_row, y2_row = min([el.y1 for el in row]), max([el.y2 for el in row])

        # Identify whitespaces that correspond vertically to rows
        row_ws = [ws for ws in whitespaces if min(ws.y2, y2_row) - max(ws.y1, y1_row) == y2_row - y1_row]

        if len([ws for ws in row_ws if min(ws.x2, x2_row) - max(ws.x1, x1_row) > 0]) > 0:
            y_top = min(y_top, y1_row)
            y_bottom = max(y_bottom, y2_row)

    # Reprocess whitespaces
    whitespaces = sorted(whitespaces, key=lambda w: w.x1 + w.x2)
    reprocessed_ws = list()
    for idx, ws in enumerate(whitespaces):
        if idx == 0:
            # Left border
            ws = Cell(x1=int(ws.x2 - char_length),
                      y1=max(ws.y1, y_top),
                      x2=int(ws.x2 - char_length),
                      y2=min(ws.y2, y_bottom))
        elif idx == len(whitespaces) - 1:
            # Right border
            ws = Cell(x1=int(ws.x1 + char_length),
                      y1=max(ws.y1, y_top),
                      x2=int(ws.x1 + char_length),
                      y2=min(ws.y2, y_bottom))
        else:
            # Column delimiters
            ws = Cell(x1=(ws.x1 + ws.x2) // 2,
                      y1=max(ws.y1, y_top),
                      x2=(ws.x1 + ws.x2) // 2,
                      y2=min(ws.y2, y_bottom))

        reprocessed_ws.append(ws)

    return reprocessed_ws


def identify_columns(table_segment: TableSegment, char_length: float) -> Optional[DelimiterGroup]:
    """
    Identify list of vertical delimiters that can be table columns in a table segment
    :param table_segment: TableSegment object
    :param char_length: average character length
    :return: delimiter group that can correspond to columns
    """
    # Get columns whitespaces
    columns = get_columns_delimiters(table_segment=table_segment,
                                     char_length=char_length)

    # Resize columns
    resized_columns = get_relevant_height(whitespaces=columns,
                                          elements=table_segment.elements,
                                          char_length=char_length)

    # Create delimiter group
    x1_del, x2_del = min([d.x1 for d in resized_columns]), max([d.x2 for d in resized_columns])
    y1_del, y2_del = min([d.y1 for d in resized_columns]), max([d.y2 for d in resized_columns])
    delimiter_group = DelimiterGroup(delimiters=resized_columns,
                                     elements=[el for el in table_segment.elements if el.x1 >= x1_del
                                               and el.x2 <= x2_del and el.y1 >= y1_del and el.y2 <= y2_del])

    return delimiter_group if len(delimiter_group.delimiters) >= 4 and len(delimiter_group.elements) > 0 else None

