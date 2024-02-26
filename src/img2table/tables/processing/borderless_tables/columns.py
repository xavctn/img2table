# coding: utf-8

import copy
from typing import Optional, List

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import TableSegment, Whitespace, Column, VerticalWS, \
    ColumnGroup


def get_columns_delimiters(table_segment: TableSegment, char_length: float) -> List[Column]:
    """
    Identify column delimiters in table segment
    :param table_segment: TableSegment object
    :param char_length: average character length
    :return: list of column delimiters
    """
    # Get whitespaces
    table_areas = sorted(table_segment.table_areas, key=lambda x: x.position)

    # Create groups of relevant vertical whitespaces
    columns = list()
    for id_area, tb_area in enumerate(table_areas):
        new_columns = list()
        whitespaces = [VerticalWS(ws=ws,
                                  top=ws.y1 == tb_area.y1,
                                  bottom=ws.y2 == tb_area.y2,
                                  position=id_area)
                       for ws in tb_area.whitespaces]

        for col in columns:
            # Get matching whitespaces
            matching_ws = [v_ws for v_ws in whitespaces if col.corresponds(v_ws=v_ws, char_length=char_length)]

            if matching_ws:
                for v_ws in matching_ws:
                    # Update whitespace
                    setattr(v_ws, "used", True)

                    # Create new column
                    new_col = copy.deepcopy(col)
                    new_col.add(v_ws)
                    new_columns.append(new_col)
            else:
                new_columns.append(col)

        # Create columns corresponding to unused whitespaces
        new_columns += [Column.from_ws(v_ws=v_ws) for v_ws in whitespaces if not v_ws.used]

        # Replace existing columns by new columns
        columns = new_columns

    # Recompute boundaries of columns (up to previous/next area)
    dict_bounds = {k: {"y_min": table_areas[k].y1, "y_max": table_areas[k].y2}
                   for k in range(len(table_areas))}
    reshaped_columns = list()
    for col in columns:
        reshaped_whitespaces = list()
        for v_ws in col.whitespaces:
            # Reshape all whitespaces
            y_min = dict_bounds.get(v_ws.position - 1, {}).get("y_max") or v_ws.y1 if v_ws.top else v_ws.y1
            y_max = dict_bounds.get(v_ws.position + 1, {}).get("y_min") or v_ws.y2 if v_ws.bottom else v_ws.y2
            reshaped_v_ws = VerticalWS(ws=Whitespace(cells=[Cell(x1=col.x1,
                                                                 y1=y_min if c.y1 == v_ws.y1 else c.y1,
                                                                 x2=col.x2,
                                                                 y2=y_max if c.y2 == v_ws.y2 else c.y2)
                                                            for c in v_ws.ws.cells]),)
            reshaped_whitespaces.append(reshaped_v_ws)

        # Create reshaped column
        reshaped_col = Column(whitespaces=reshaped_whitespaces)
        reshaped_columns.append(reshaped_col)

    # Keep only columns that represent at least 66% of the maximum height
    max_height = max(map(lambda col: col.height, reshaped_columns))
    reshaped_columns = [col for col in reshaped_columns if col.height >= 0.66 * max_height]

    return reshaped_columns


def get_relevant_height(columns: List[Column], elements: List[Cell], char_length: float,
                        median_line_sep: float) -> List[Column]:
    """
    Get relevant columns height relative to image elements
    :param columns: list of column delimiters
    :param elements: list of image elements
    :param char_length: average character length
    :param median_line_sep: median line separation
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
    y_top, y_bottom, = max([col.y2 for col in columns]), min([col.y1 for col in columns])
    for row in rows:
        x1_row, x2_row = min([el.x1 for el in row]), max([el.x2 for el in row])
        y1_row, y2_row = min([el.y1 for el in row]), max([el.y2 for el in row])

        # Identify columns that correspond vertically to rows
        row_cols = [col for col in columns if min(col.y2, y2_row) - max(col.y1, y1_row) == y2_row - y1_row]

        if len([col for col in row_cols if min(col.x2, x2_row) - max(col.x1, x1_row) > 0]) > 0:
            y_top = min(y_top, y1_row)
            y_bottom = max(y_bottom, y2_row)

    # Reprocess columns
    columns = sorted(columns, key=lambda col: col.x1 + col.x2)
    reprocessed_cols = list()
    for idx, col in enumerate(columns):
        if idx == 0:
            # Left border
            new_v_ws = list()
            for v_ws in col.whitespaces:
                ws_cells = [Cell(x1=c.x2,
                                 y1=y_top - int(0.5 * char_length) if c.y1 == y_top else max(c.y1, y_top),
                                 x2=c.x2,
                                 y2=y_bottom + int(0.5 * char_length) if c.y2 == y_bottom else min(c.y2, y_bottom))
                            for c in v_ws.ws.cells
                            if min(c.y2, y_bottom) - max(c.y1, y_top) >= min(median_line_sep, v_ws.height)]
                if len(ws_cells) > 0:
                    new_v_ws.append(VerticalWS(ws=Whitespace(cells=ws_cells)))

        elif idx == len(columns) - 1:
            # Right border
            new_v_ws = list()
            for v_ws in col.whitespaces:
                ws_cells = [Cell(x1=c.x1,
                                 y1=y_top - int(0.5 * char_length) if c.y1 == y_top else max(c.y1, y_top),
                                 x2=c.x1,
                                 y2=y_bottom + int(0.5 * char_length) if c.y2 == y_bottom else min(c.y2, y_bottom))
                            for c in v_ws.ws.cells
                            if min(c.y2, y_bottom) - max(c.y1, y_top) >= min(median_line_sep, v_ws.height)]
                if len(ws_cells) > 0:
                    new_v_ws.append(VerticalWS(ws=Whitespace(cells=ws_cells)))
        else:
            # Column delimiters
            new_v_ws = list()
            for v_ws in col.whitespaces:
                ws_cells = [Cell(x1=(c.x1 + c.x2) // 2,
                                 y1=y_top - int(0.5 * char_length) if c.y1 == y_top else max(c.y1, y_top),
                                 x2=(c.x1 + c.x2) // 2,
                                 y2=y_bottom + int(0.5 * char_length) if c.y2 == y_bottom else min(c.y2, y_bottom))
                            for c in v_ws.ws.cells
                            if min(c.y2, y_bottom) - max(c.y1, y_top) >= min(median_line_sep, v_ws.height)]
                if len(ws_cells) > 0:
                    new_v_ws.append(VerticalWS(ws=Whitespace(cells=ws_cells)))

        if len(new_v_ws) > 0:
            reprocessed_cols.append(Column(whitespaces=new_v_ws))

    if reprocessed_cols:
        # Keep only columns that represent at least 66% of the maximum height
        max_height = max(map(lambda col: col.height, reprocessed_cols))
        reprocessed_cols = [col for col in reprocessed_cols if col.height >= 0.66 * max_height]

        return reprocessed_cols
    else:
        return []


def identify_columns(table_segment: TableSegment, char_length: float, median_line_sep: float) -> Optional[ColumnGroup]:
    """
    Identify list of vertical delimiters that can be table columns in a table segment
    :param table_segment: TableSegment object
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: delimiter group that can correspond to columns
    """
    # Get columns whitespaces
    columns = get_columns_delimiters(table_segment=table_segment,
                                     char_length=char_length)

    # Resize columns
    resized_columns = get_relevant_height(columns=columns,
                                          elements=table_segment.elements,
                                          char_length=char_length,
                                          median_line_sep=median_line_sep)

    if resized_columns:
        # Create column group
        x1_del, x2_del = min([d.x1 for d in resized_columns]), max([d.x2 for d in resized_columns])
        y1_del, y2_del = min([d.y1 for d in resized_columns]), max([d.y2 for d in resized_columns])
        column_group = ColumnGroup(columns=resized_columns,
                                   elements=[el for el in table_segment.elements if el.x1 >= x1_del
                                             and el.x2 <= x2_del and el.y1 >= y1_del and el.y2 <= y2_del],
                                   char_length=char_length)

        return column_group if len(column_group.columns) >= 4 and len(column_group.elements) > 0 else None

    return None

