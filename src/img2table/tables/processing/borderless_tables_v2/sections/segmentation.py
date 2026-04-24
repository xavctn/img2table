import numpy as np

from img2table.tables.processing.borderless_tables_v2._model import (
    ColumnSection,
    MergedRow,
    RowCharacteristic,
    Whitespace,
)


def matching_whitespaces(
    ws1_list: list[Whitespace], ws2_list: list[Whitespace], min_width: float
) -> tuple[bool, list[Whitespace]]:
    """
    Identify if both sets of whitespaces match
    :param ws1_list: first set of whitespaces
    :param ws2_list: second set of whitespaces
    :param min_width: minimum column width
    :return: boolean indicating whether two sets of whitespaces match and resultant whitespaces
    """

    # Get largest and smallest list of whitespaces and iterate over the shortest list
    ws_short, ws_long = (
        (ws1_list, ws2_list) if len(ws1_list) <= len(ws2_list) else (ws2_list, ws1_list)
    )

    matching_ws, covered_long_indices = [], set()
    for ws_s in ws_short:
        found_matching_ws = False
        for idx, ws_l in enumerate(ws_long):
            # Compute overlap
            overlap = min(ws_s.end, ws_l.end) - max(ws_s.start, ws_l.start)

            # Check overlap is sufficient or if bounds match
            if overlap >= max(
                0.5 * min(ws_s.width, ws_l.width, 5 * min_width), 0.5 * min_width
            ) or ws_s.matching_bound(ws_l):
                matching_ws.append(
                    Whitespace(
                        start=max(ws_s.start, ws_l.start),
                        end=min(ws_s.end, ws_l.end),
                        start_bound=ws_s.start_bound and ws_l.start_bound,
                        end_bound=ws_s.end_bound and ws_l.end_bound,
                    )
                )
                covered_long_indices.add(idx)
                found_matching_ws = True

        if not found_matching_ws:
            return False, []

    if len(covered_long_indices) < len(ws_long):
        return False, []

    # Check that content overlaps (based on whitespaces)
    ws1_min, ws1_max = min(ws.end for ws in ws1_list), max(ws.start for ws in ws1_list)
    ws2_min, ws2_max = min(ws.end for ws in ws2_list), max(ws.start for ws in ws2_list)
    if min(ws1_max, ws2_max) - max(ws1_min, ws2_min) < min_width:
        return False, sorted(matching_ws, key=lambda x: x.start)

    return True, sorted(matching_ws, key=lambda x: x.start)


def _row_group_score(
    row_data: list[RowCharacteristic], index: int, max_gap: float, min_width: float
) -> int:
    """
    Count consecutive rows above and below `index` that have matching whitespaces.
    :param row_data: list of all row characteristics
    :param index: index of the candidate seed row
    :param max_gap: maximum vertical gap between consecutive rows
    :param min_width: minimum whitespace width for matching
    :return: number of consecutive matching neighbors in both directions
    """
    count = 0
    current_ws = row_data[index].ws
    for i in range(index + 1, len(row_data)):
        if abs(row_data[i].row.y_center - row_data[i - 1].row.y_center) > max_gap:
            break
        is_match, current_ws = matching_whitespaces(current_ws, row_data[i].ws, min_width)
        if not is_match:
            break
        count += 1

    current_ws = row_data[index].ws
    for i in range(index - 1, -1, -1):
        if abs(row_data[i + 1].row.y_center - row_data[i].row.y_center) > max_gap:
            break
        is_match, current_ws = matching_whitespaces(current_ws, row_data[i].ws, min_width)
        if not is_match:
            break
        count += 1

    return count


def compute_column_section(
    merged_rows: list[MergedRow],
    min_width: float,
    x_min: int,
    x_max: int,
    ratio_vertical_separation: float,
) -> tuple[list[ColumnSection], float]:
    """
    Compute column sections from merged rows.
    :param merged_rows: list of merged rows
    :param min_width: minimum width for a whitespace to be considered
    :param x_min: start of span
    :param x_max: end of span
    :param ratio_vertical_separation: ratio of median row separation to use as max vertical separation
    :return: list of column sections and maximum gap between rows
    """
    # Compute median row separation on overlapping rows
    row_separations: list[float] = []
    for idx, row in enumerate(merged_rows[:-1]):
        for other_row in merged_rows[idx + 1 :]:
            # Check if rows overlap
            overlap = min(row.x2, other_row.x2) - max(row.x1, other_row.x1) >= min_width
            if overlap:
                row_separations.append(other_row.y_center - row.y_center)
                break
    median_row_separation = np.median(row_separations) if row_separations else 0
    max_gap = float(median_row_separation * ratio_vertical_separation)

    # Compute all rows characteristics
    row_data = [
        RowCharacteristic(
            index=idx,
            row=row,
            ws=row.compute_whitespaces(min_width=min_width, x_min=x_min, x_max=x_max),
        )
        for idx, row in enumerate(merged_rows)
    ]

    # Pre-compute group scores: how many consecutive matching rows each row has above/below.
    group_scores = [_row_group_score(row_data, rd.index, max_gap, min_width) for rd in row_data]

    # Construct column sections by starting with the row with the most whitespaces and expanding upwards/downwards
    used_rows: set[int] = set()
    column_sections: list[ColumnSection] = []
    while len(used_rows) < len(merged_rows):
        # Check available rows
        available = [r for r in row_data if r.index not in used_rows]
        if not available:
            break

        # Get the row most likely to be the core of a table.
        seed = max(
            available,
            key=lambda x: ((x.count - 1) * (group_scores[x.index] + 1), x.inner_ws_width),
        )
        section = ColumnSection().update(row=seed.row, whitespaces=seed.ws)
        used_rows.add(seed.index)

        # Downwards expansion
        for row_idx in range(seed.index + 1, len(merged_rows)):
            if row_idx in used_rows:
                break
            target = row_data[row_idx]

            # Check vertical gap
            if (gap := abs(section.last_y_center - target.row.y_center)) > max_gap:
                break

            # Check whitespace correspondence
            is_match, match_ws = matching_whitespaces(section.whitespaces, target.ws, min_width)
            if is_match or (gap < 0.66 * median_row_separation and match_ws):
                section.update(row=target.row, whitespaces=match_ws)
                used_rows.add(row_idx)
            else:
                break

        # Upwards expansion
        for row_idx in range(seed.index - 1, -1, -1):
            if row_idx in used_rows:
                break
            target = row_data[row_idx]

            # Check vertical gap
            if (gap := abs(section.first_y_center - target.row.y_center)) > max_gap:
                break

            # Check whitespace correspondence
            is_match, match_ws = matching_whitespaces(section.whitespaces, target.ws, min_width)
            if is_match or (gap < 0.66 * median_row_separation and match_ws):
                section.update(row=target.row, whitespaces=match_ws)
                used_rows.add(row_idx)
            else:
                break

        column_sections.append(section)

    column_sections.sort(key=lambda sec: sec.y1)

    return column_sections, max_gap
