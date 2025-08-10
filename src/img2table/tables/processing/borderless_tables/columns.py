
import copy
from typing import Optional

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import TableSegment, Whitespace, Column, VerticalWS, \
    ColumnGroup


def get_columns_delimiters(table_segment: TableSegment, char_length: float) -> list[Column]:
    """
    Identify column delimiters in table segment
    :param table_segment: TableSegment object
    :param char_length: average character length
    :return: list of column delimiters
    """
    # Get whitespaces
    table_areas = sorted(table_segment.table_areas, key=lambda x: x.position)

    # Create groups of relevant vertical whitespaces
    columns = []
    for id_area, tb_area in enumerate(table_areas):
        new_columns = []
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
                    v_ws.used = True

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
    reshaped_columns = []
    for col in columns:
        reshaped_whitespaces = []
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
    return [col for col in reshaped_columns if col.height >= 0.66 * max_height]


def identify_columns(table_segment: TableSegment, char_length: float) -> Optional[ColumnGroup]:
    """
    Identify list of vertical delimiters that can be table columns in a table segment
    :param table_segment: TableSegment object
    :param char_length: average character length
    :return: delimiter group that can correspond to columns
    """
    # Get columns whitespaces
    columns = get_columns_delimiters(table_segment=table_segment,
                                     char_length=char_length)

    if columns:
        # Create column group
        x1_del, x2_del = min([d.x1 for d in columns]), max([d.x2 for d in columns])
        y1_del, y2_del = min([d.y1 for d in columns]), max([d.y2 for d in columns])
        column_group = ColumnGroup(columns=columns,
                                   elements=[el for el in table_segment.elements if el.x1 >= x1_del
                                             and el.x2 <= x2_del and el.y1 >= y1_del and el.y2 <= y2_del],
                                   char_length=char_length)

        return column_group if len(column_group.columns) >= 4 and len(column_group.elements) > 0 else None

    return None

