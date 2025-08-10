

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.cells import get_cells
from img2table.tables.processing.bordered_tables.tables import cluster_to_table
from img2table.tables.processing.borderless_tables.model import ImageSegment, Whitespace
from img2table.tables.processing.borderless_tables.whitespaces import get_whitespaces


def implicit_rows_lines(table: Table, segment: ImageSegment) -> list[Line]:
    """
    Identify lines corresponding to implicit rows
    :param table: table
    :param segment: ImageSegment used for whitespaces computation
    :return: list of lines corresponding to implicit rows
    """
    # Horizontal whitespaces
    h_ws = get_whitespaces(segment=segment,
                           vertical=False,
                           pct=1)

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

    # Identify relevant whitespace height
    if len(h_ws) > 2:
        full_ws_h = sorted([ws.height for ws in h_ws[1:-1] if ws.width == max([w.width for w in h_ws])])
        min_height = 0.5 * full_ws_h[len(full_ws_h) // 2 + len(full_ws_h) % 2 - 1] if len(full_ws_h) >= 3 else 1
        h_ws = [h_ws[0]] + [ws for ws in h_ws[1:-1] if ws.height >= min_height] + [h_ws[-1]]

    # Identify created lines
    created_lines = []
    for ws in h_ws:
        if not any(line for line in table.lines if ws.y1 <= line.y1 <= ws.y2 and line.horizontal):
            created_lines.append(Line(x1=table.x1,
                                      y1=(ws.y1 + ws.y2) // 2,
                                      x2=table.x2,
                                      y2=(ws.y1 + ws.y2) // 2))

    return created_lines


def implicit_columns_lines(table: Table, segment: ImageSegment, char_length: float) -> list[Line]:
    """
    Identify lines corresponding to implicit columns
    :param table: table
    :param segment: ImageSegment used for whitespaces computation
    :param char_length: average character length
    :return: list of lines corresponding to implicit columns
    """
    # Vertical whitespaces
    v_ws = get_whitespaces(segment=segment,
                           vertical=True,
                           min_width=char_length,
                           pct=1)

    # Identify created lines
    created_lines = []
    for ws in v_ws:
        if not any(line for line in table.lines if ws.x1 <= line.x1 <= ws.x2 and line.vertical):
            created_lines.append(Line(x1=(ws.x1 + ws.x2) // 2,
                                      y1=table.y1,
                                      x2=(ws.x1 + ws.x2) // 2,
                                      y2=table.y2))

    return created_lines


def implicit_content(table: Table, contours: list[Cell], char_length: float, implicit_rows: bool = False,
                     implicit_columns: bool = False) -> Table:
    """
    Identify implicit content in table
    :param table: Table object
    :param contours: image contours
    :param char_length: average character length
    :param implicit_rows: boolean indicating if implicit rows should be detected
    :param implicit_columns: boolean indicating if implicit columns should be detected
    :return: Table with implicit content detected
    """
    if not implicit_rows and not implicit_columns:
        return table

    # Get table contours and create corresponding segment
    tb_contours = [c for c in contours
                   if c.x1 >= table.x1 and c.x2 <= table.x2
                   and c.y1 >= table.y1 and c.y2 <= table.y2]
    segment = ImageSegment(x1=table.x1, y1=table.y1, x2=table.x2, y2=table.y2,
                           elements=tb_contours)

    # Create new lines
    lines = table.lines
    if implicit_rows:
        lines += implicit_rows_lines(table=table, segment=segment)
    if implicit_columns:
        lines += implicit_columns_lines(table=table, segment=segment, char_length=char_length)

    # Create
    cells = get_cells(horizontal_lines=[line for line in lines if line.horizontal],
                      vertical_lines=[line for line in lines if line.vertical])

    return cluster_to_table(cluster_cells=cells, elements=tb_contours, borderless=False)
