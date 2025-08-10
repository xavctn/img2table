
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import ColumnGroup, Whitespace
from img2table.tables.processing.borderless_tables.whitespaces import get_whitespaces


def identify_row_delimiters(column_group: ColumnGroup) -> list[Cell]:
    """
    Identify list of rows corresponding to the delimiter group
    :param column_group: column delimiters group
    :return: list of rows delimiters corresponding to the delimiter group
    """
    # Identify vertical whitespaces
    h_ws = get_whitespaces(segment=column_group, vertical=False, pct=0.66)

    # Create whitespaces at the top or the bottom if they are missing
    if h_ws[0].y1 > column_group.y1:
        up_ws = Whitespace(cells=[Cell(x1=min([ws.x1 for ws in h_ws]),
                                       x2=max([ws.x2 for ws in h_ws]),
                                       y1=column_group.y1,
                                       y2=min([el.y1 for el in column_group.elements]))])
        h_ws.insert(0, up_ws)

    if h_ws[-1].y2 < column_group.y2:
        down_ws = Whitespace(cells=[Cell(x1=min([ws.x1 for ws in h_ws]),
                                         x2=max([ws.x2 for ws in h_ws]),
                                         y1=column_group.y2,
                                         y2=max([el.y2 for el in column_group.elements]))])
        h_ws.append(down_ws)

    # Identify relevant whitespace height
    if len(h_ws) > 2:
        full_ws_h = sorted([ws.height for ws in h_ws[1:-1] if ws.width == max([w.width for w in h_ws])])
        min_height = 0.5 * full_ws_h[len(full_ws_h) // 2 + len(full_ws_h) % 2 - 1] if len(full_ws_h) >= 3 else 1
        h_ws = [h_ws[0]] + [ws for ws in h_ws[1:-1] if ws.height >= min_height] + [h_ws[-1]]

    # Filter relevant whitespaces
    deleted_idx = []
    for i in range(len(h_ws)):
        for j in range(i, len(h_ws)):
            # Check if both whitespaces are adjacent
            adjacent = len({h_ws[i].y1, h_ws[i].y2}.intersection({h_ws[j].y1, h_ws[j].y2})) > 0

            if adjacent:
                if h_ws[i].width > h_ws[j].width:
                    deleted_idx.append(j)
                elif h_ws[i].width < h_ws[j].width:
                    deleted_idx.append(i)

    h_ws = [ws for idx, ws in enumerate(h_ws) if idx not in deleted_idx]

    # Create delimiters
    final_delims = []
    for ws in h_ws:
        if ws.y1 == column_group.y1 or ws.y2 == column_group.y2:
            continue

        final_delims.append(Cell(x1=ws.x1,
                                 x2=ws.x2,
                                 y1=(ws.y1 + ws.y2) // 2,
                                 y2=(ws.y1 + ws.y2) // 2))

    # Add top and bottom row delimiters
    x1_els, x2_els = min([el.x1 for el in column_group.elements]), max([el.x2 for el in column_group.elements])
    y1_els, y2_els = min([el.y1 for el in column_group.elements]), max([el.y2 for el in column_group.elements])
    final_delims += [Cell(x1=x1_els, x2=x2_els, y1=y1_els, y2=y1_els),
                     Cell(x1=x1_els, x2=x2_els, y1=y2_els, y2=y2_els)]

    return sorted(final_delims, key=lambda d: d.y1)


def filter_coherent_row_delimiters(row_delimiters: list[Cell], column_group: ColumnGroup) -> list[Cell]:
    """
    Filter coherent row delimiters (i.e that properly delimit relevant text)
    :param row_delimiters: list of row delimiters
    :param column_group: column delimiters group
    :return: filtered row delimiters
    """
    # Get max width of delimiters
    max_width = max(map(lambda d: d.width, row_delimiters))

    delimiters_to_delete = []
    for idx, delim in enumerate(row_delimiters):
        if delim.width >= 0.95 * max_width:
            continue

        # Get area above delimiter and corresponding columns
        upper_delim = row_delimiters[idx - 1]
        upper_area = Cell(x1=max(delim.x1, upper_delim.x1),
                          y1=upper_delim.y2,
                          x2=min(delim.x2, upper_delim.x2),
                          y2=delim.y1)
        upper_columns = sorted([col for col in column_group.columns
                                if min(upper_area.y2, col.y2) - max(upper_area.y1, col.y1) >= 0.8 * upper_area.height
                                and upper_area.x1 <= col.x1 <= upper_area.x2],
                               key=lambda c: c.x1)
        # Get contained elements in upper area
        upper_contained_elements = [el for el in column_group.elements if el.y1 >= upper_area.y1
                                    and el.y2 <= upper_area.y2 and el.x1 >= upper_columns[0].x2
                                    and el.x2 <= upper_columns[-1].x1] if upper_columns else []

        # Get area below delimiter and corresponding columns
        bottom_delim = row_delimiters[idx + 1]
        bottom_area = Cell(x1=max(delim.x1, bottom_delim.x1),
                           y1=delim.y2,
                           x2=min(delim.x2, bottom_delim.x2),
                           y2=bottom_delim.y1)
        bottom_columns = sorted([col for col in column_group.columns
                                 if min(bottom_area.y2, col.y2) - max(bottom_area.y1, col.y1) >= 0.8 * bottom_area.height
                                 and bottom_area.x1 <= col.x1 <= bottom_area.x2],
                                key=lambda c: c.x1)
        # Get contained elements in bottom area
        bottom_contained_elements = [el for el in column_group.elements if el.y1 >= bottom_area.y1
                                     and el.y2 <= bottom_area.y2 and el.x1 >= bottom_columns[0].x2
                                     and el.x2 <= bottom_columns[-1].x1] if bottom_columns else []

        # If one of the area is empty, the delimiter is irrelevant
        if len(upper_contained_elements) * len(bottom_contained_elements) == 0:
            delimiters_to_delete.append(idx)

    return [d for idx, d in enumerate(row_delimiters) if idx not in delimiters_to_delete]


def correct_delimiter_width(row_delimiters: list[Cell], contours: list[Cell]) -> list[Cell]:
    """
    Correct delimiter width if needed
    :param row_delimiters: list of row delimiters
    :param contours: list of image contours
    :return: list of row delimiters with corrected width
    """
    x_min, x_max = min([d.x1 for d in row_delimiters]), max([d.x2 for d in row_delimiters])

    for idx, delim in enumerate(row_delimiters):
        if delim.width == x_max - x_min:
            continue

        # Check if there are contours on the left of the delimiter
        left_contours = [c for c in contours if c.y1 + c.height // 6 < delim.y1 < c.y2 - c.height // 6
                         and min(c.x2, delim.x1) - max(c.x1, x_min) > 0]
        delim_x_min = max([c.x2 for c in left_contours] + [x_min])

        # Check if there are contours on the right of the delimiter
        right_contours = [c for c in contours if c.y1 + c.height // 6 < delim.y1 < c.y2 - c.height // 6
                          and min(c.x2, x_max) - max(c.x1, delim.x2) > 0]
        delim_x_max = min([c.x1 for c in right_contours] + [x_max])

        # Update delimiter width
        row_delimiters[idx].x1 = delim_x_min
        row_delimiters[idx].x2 = delim_x_max

    return row_delimiters


def identify_delimiter_group_rows(column_group: ColumnGroup, contours: list[Cell]) -> list[Cell]:
    """
    Identify list of rows corresponding to the delimiter group
    :param column_group: column delimiters group
    :param contours: list of image contours
    :return: list of rows delimiters corresponding to the delimiter group
    """
    # Get row delimiters
    row_delimiters = identify_row_delimiters(column_group=column_group)

    if row_delimiters:
        # Filter coherent delimiters
        coherent_delimiters = filter_coherent_row_delimiters(row_delimiters=row_delimiters,
                                                             column_group=column_group)

        # Correct delimiters width
        corrected_delimiters = correct_delimiter_width(row_delimiters=coherent_delimiters,
                                                       contours=contours)

        return corrected_delimiters if len(corrected_delimiters) >= 3 else []
    return []

