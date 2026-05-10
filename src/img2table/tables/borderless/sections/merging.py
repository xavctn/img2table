from collections import defaultdict

import numpy as np

from img2table.tables.borderless.types import (
    ColumnSection,
    Whitespace,
    compute_whitespaces,
)


def assess_table_whitespace_coherency(
    list_ws_1: list[Whitespace],
    list_ws_2: list[Whitespace],
    min_width: float,
    vertically_close: bool,
) -> bool:
    """
    Assess whether the whitespaces of 2 sections are coherent.
    :param list_ws_1: Set of whitespaces from first section.
    :param list_ws_2: Set of whitespaces from second section.
    :param min_width: Minimum width for a whitespace to be considered significant.
    :param vertically_close: Whether the two sections are vertically close.
    :return: True if the whitespaces are coherent, False otherwise.
    """
    # Single column case
    if min(len(list_ws_1), len(list_ws_2)) < 3:
        return vertically_close & (len(list_ws_1) == len(list_ws_2))

    # Quick same columns check - all matches
    if len(list_ws_1) == len(list_ws_2) and all(
        min(ws1.end, ws2.end) - max(ws1.start, ws2.start) >= 0.5 * min_width
        for ws1, ws2 in zip(list_ws_1, list_ws_2, strict=True)
    ):
        return True

    ws_short, ws_long = (
        (list_ws_1, list_ws_2) if len(list_ws_1) < len(list_ws_2) else (list_ws_2, list_ws_1)
    )
    # Check whitespaces coherency on "middle" whitespaces
    ws_overlap_width = sum(
        max(
            (
                overlap
                for ws_s in ws_short
                if (overlap := max(0, min(ws_l.end, ws_s.end) - max(ws_l.start, ws_s.start)))
                > 0.5 * min(ws_l.width, ws_s.width)
            ),
            default=-ws_l.width / 2,
        )
        for ws_l in ws_long
    )

    # Compute threshold for coherency
    if min(len(list_ws_1), len(list_ws_2)) < 4:
        threshold = 1
    elif vertically_close:
        threshold = 0.66
    else:
        threshold = 0.8

    # Compute coherency threshold
    return ws_overlap_width / sum(ws.width for ws in ws_long) >= threshold


def _flush_sections_group(
    group: list[ColumnSection], common_ws: list[Whitespace], x_min: int, x_max: int
) -> list[ColumnSection]:
    """
    Creates a single merged ColumnSection if the group has 2+ elements.
    :param group: List of column sections to merge.
    :param common_ws: List of common whitespaces between the sections.
    :param x_min: start of span
    :param x_max: end of span
    :return: Merged list of column sections.
    """
    if len(group) < 2:
        return group

    # Compute whitespaces for group items and keep only ones matching with common_ws
    items = [it for s in group for it in s.items]
    whitespaces = compute_whitespaces(items=items, min_width=1, x_min=x_min, x_max=x_max)
    filtered_ws = [
        ws
        for ws in whitespaces
        if any(min(ws.end, cws.end) > max(ws.start, cws.start) for cws in common_ws)
    ]

    return [
        ColumnSection(
            items=items, rows=[row for s in group for row in s.rows], whitespaces=filtered_ws
        )
    ]


def _merge_section_whitespaces(
    sections: list[ColumnSection], min_width: float, x_min: int, x_max: int
) -> list[ColumnSection]:
    """
    Try merging consecutive column sections that have overlapping whitespaces.
    :param sections: List of column sections to merge.
    :param min_width: Minimum width of a whitespace to be considered valid.
    :param x_min: start of span
    :param x_max: end of span
    :return: Merged list of column sections.
    """
    if len(sections) <= 1:
        return sections

    # Compute total number of rows in all sections
    total_rows = sum(len(section.rows) for section in sections)

    # Sort all whitespaces by row coverage so the most common slots are evaluated first.
    def _row_coverage(ref_ws: Whitespace) -> float:
        total = 0
        for section in sections:
            coverage = max(
                min(ref_ws.end, ws.end) - max(ref_ws.start, ws.start) for ws in section.whitespaces
            )
            if coverage > 0:
                total += coverage * len(section.rows)
        return total

    all_candidates = sorted(
        {ws for section in sections for ws in section.whitespaces},
        key=_row_coverage,
        reverse=True,
    )
    all_candidates = [
        *compute_whitespaces(
            items=[it for section in sections for it in section.items],
            min_width=0.5 * min_width,
            x_min=x_min,
            x_max=x_max,
        ),
        *all_candidates,
    ]

    # Initialize common whitespaces and map of used whitespaces per section
    common_ws, map_used_ws = [], defaultdict(set)

    # Check all candidates against common whitespaces and find matching section whitespaces
    for ref_ws in all_candidates:
        if any(min(ref_ws.end, c.end) > max(ref_ws.start, c.start) for c in common_ws):
            continue
        # Find the matching whitespace in each section
        matched, matched_rows, used_ws = [], 0, defaultdict(set)
        for idx_section, section in enumerate(sections):
            for idx_ws, section_ws in enumerate(section.whitespaces):
                m_start = max(ref_ws.start, section_ws.start)
                m_end = min(ref_ws.end, section_ws.end)
                if m_end > m_start:
                    matched.append(
                        Whitespace(
                            start=m_start,
                            end=m_end,
                            start_bound=ref_ws.start_bound and section_ws.start_bound,
                            end_bound=ref_ws.end_bound and section_ws.end_bound,
                        )
                    )
                    if idx_ws not in map_used_ws[idx_section]:
                        matched_rows += len(section.rows)
                    used_ws[idx_section].add(idx_ws)
                    break

        # Keep the whitespace if it is present in a majority of rows
        if matched_rows >= 0.75 * total_rows:
            final = Whitespace(
                start=int(np.median([m.start for m in matched])),
                end=int(np.median([m.end for m in matched])),
                start_bound=all(m.start_bound for m in matched),
                end_bound=all(m.end_bound for m in matched),
            )
            if final.width >= 0.5 * min_width:
                common_ws.append(final)
                for idx_section, vals in used_ws.items():
                    map_used_ws[idx_section].update(vals)

    if not common_ws:
        return sections

    # Only merge sections that share all majority whitespaces; keep others as singular
    def shares_majority(section: ColumnSection) -> bool:
        return all(
            any(
                min(ws.end, section_ws.end) > max(ws.start, section_ws.start)
                for section_ws in section.whitespaces
            )
            for ws in common_ws
        )

    # Create merged sections
    result = []
    current_group = []
    for section in sections:
        if shares_majority(section):
            # Add to current group for merging
            current_group.append(section)
        else:
            # Flush current group and add non-mergeable section
            result += _flush_sections_group(
                group=current_group, common_ws=common_ws, x_min=x_min, x_max=x_max
            )
            result.append(section)
            current_group = []

    # Flush remaining sections
    result += _flush_sections_group(
        group=current_group, common_ws=common_ws, x_min=x_min, x_max=x_max
    )

    return result


def merge_column_sections(
    column_sections: list[ColumnSection], min_width: float, max_gap: float, x_min: int, x_max: int
) -> list[ColumnSection]:
    """
    Merge consecutive column sections that are close vertically if their whitespaces overlap.
    :param column_sections: List of column sections to merge.
    :param min_width: Minimum width of a whitespace to be considered valid.
    :param max_gap: Maximum vertical gap between sections to consider them close.
    :param x_min: start of span
    :param x_max: end of span
    :return: Merged list of column sections.
    """
    while True:
        # Identify first section index with at least 2 columns
        first_section_with_columns = next(
            (idx for idx, sec in enumerate(column_sections) if sec.nb_columns >= 2), None
        )
        if first_section_with_columns is None:
            return column_sections

        # Create groups of columns that are close vertically
        merged_sections: list[ColumnSection] = column_sections[:first_section_with_columns]
        current_group: list[ColumnSection] = [column_sections[first_section_with_columns]]
        for section in column_sections[first_section_with_columns + 1 :]:
            if section.nb_columns < 2:
                # Flush current group
                merged_sections += _merge_section_whitespaces(
                    sections=current_group, min_width=min_width, x_min=x_min, x_max=x_max
                )
                merged_sections.append(section)
                current_group = []
            elif len(current_group) == 0:
                current_group = [section]
            elif section.first_y_center - current_group[-1].last_y_center > max_gap:
                # Flush current group
                merged_sections += _merge_section_whitespaces(
                    sections=current_group, min_width=min_width, x_min=x_min, x_max=x_max
                )
                current_group = [section]
            else:
                # Check coherency of whitespaces with previous section of the current group
                coherent = assess_table_whitespace_coherency(
                    list_ws_1=current_group[-1].whitespaces,
                    list_ws_2=section.whitespaces,
                    min_width=min_width,
                    vertically_close=(
                        section.first_y_center - current_group[-1].last_y_center <= 0.5 * max_gap
                    ),
                )
                if coherent:
                    current_group.append(section)
                else:
                    # Flush current group and start a new one
                    merged_sections += _merge_section_whitespaces(
                        sections=current_group, min_width=min_width, x_min=x_min, x_max=x_max
                    )
                    current_group = [section]

        # Flush remaining group
        merged_sections += _merge_section_whitespaces(
            sections=current_group, min_width=min_width, x_min=x_min, x_max=x_max
        )

        if len(merged_sections) == len(column_sections):
            return merged_sections
        column_sections = merged_sections
