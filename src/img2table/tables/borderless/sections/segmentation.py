from itertools import pairwise

import numpy as np

from img2table.tables.borderless.types import (
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
    ws_short, ws_long = (
        (ws1_list, ws2_list) if len(ws1_list) <= len(ws2_list) else (ws2_list, ws1_list)
    )

    matching_ws: list[Whitespace] = []
    covered_long = [False] * len(ws_long)
    long_idx = 0

    for ws_s in ws_short:
        # Skip whitespaces ending before the current whitespace starts
        while long_idx < len(ws_long) and ws_long[long_idx].end < ws_s.start:
            long_idx += 1

        scan_idx = long_idx
        found_matching_ws = False
        while scan_idx < len(ws_long) and ws_long[scan_idx].start <= ws_s.end:
            ws_l = ws_long[scan_idx]

            # Compute overlap
            overlap = min(ws_s.end, ws_l.end) - max(ws_s.start, ws_l.start)

            # Check overlap is sufficient or if bounds match
            if overlap >= max(
                0.5 * min(ws_s.width, ws_l.width, 3 * min_width), 0.5 * min_width
            ) or ws_s.matching_bound(ws_l):
                matching_ws.append(
                    Whitespace(
                        start=max(ws_s.start, ws_l.start),
                        end=min(ws_s.end, ws_l.end),
                        start_bound=ws_s.start_bound and ws_l.start_bound,
                        end_bound=ws_s.end_bound and ws_l.end_bound,
                    )
                )
                covered_long[scan_idx] = True
                found_matching_ws = True
            scan_idx += 1

        if not found_matching_ws:
            return False, []

    if not all(covered_long):
        return False, []

    # Check that content overlaps (based on whitespaces)
    ws1_min, ws1_max = ws1_list[0].end, ws1_list[-1].start
    ws2_min, ws2_max = ws2_list[0].end, ws2_list[-1].start
    if min(ws1_max, ws2_max) - max(ws1_min, ws2_min) < min_width:
        return False, matching_ws

    return True, matching_ws


def score_rows(row_data: list[RowCharacteristic], min_width: float, max_gap: float) -> list[int]:
    """
    Score rows based on adjacent matching whitespaces.
    :param row_data: list of row characteristics
    :param min_width: minimum whitespace width for matching
    :param max_gap: maximum vertical gap between consecutive rows
    :return: list with scores associated to row indices
    """
    # Compute if adjacent rows have matching whitespaces
    adjacent_matches: list[bool] = []
    for prv_row, nxt_row in pairwise(row_data):
        if nxt_row.row.y_center - prv_row.row.y_center > max_gap:
            adjacent_matches.append(False)
        else:
            match, _ = matching_whitespaces(prv_row.ws, nxt_row.ws, min_width)
            adjacent_matches.append(match)

    # Score rows from top to bottom based on adjacent matches
    down_scores = [0] * len(row_data)
    for idx in range(len(row_data) - 2, -1, -1):
        if adjacent_matches[idx]:
            down_scores[idx] = down_scores[idx + 1] + 1

    # Score rows from bottom to top based on adjacent matches
    up_scores = [0] * len(row_data)
    for idx in range(1, len(row_data)):
        if adjacent_matches[idx - 1]:
            up_scores[idx] = up_scores[idx - 1] + 1

    return [up_scores[idx] + down_scores[idx] for idx in range(len(row_data))]


def compute_column_sections(
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

    # Score rows for seed selection
    group_scores = score_rows(row_data=row_data, min_width=min_width, max_gap=max_gap)

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
