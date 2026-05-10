from itertools import pairwise

import numpy as np

from img2table.tables.borderless.types import (
    ColumnSection,
    MergedRow,
    Whitespace,
    compute_whitespaces,
    identify_merged_rows,
)


def ensure_section_bounds_consistency(
    section: ColumnSection, min_width: float, x_min: int, x_max: int
) -> list[ColumnSection]:
    """
    Check that top / bottom elements of the column section are consistent with the rest of the section
    :param section: Column section to check.
    :param min_width: minimum width for a whitespace to be considered
    :param x_min: start of span
    :param x_max: end of span
    :return: List of consistent column sections.
    """
    if section.nb_columns < 2 or len(section.rows) < 2:
        return [section]

    # Identify core elements (with at least 2 columns)
    rows: list[MergedRow] = sorted(section.rows, key=lambda x: x.y1)
    core_indices = [
        i for i, r in enumerate(rows) if len(r.compute_whitespaces(min_width, x_min, x_max)) > 2
    ]

    # Only one row with columns: split the section into multiple sections
    if len(core_indices) < 2:
        return [
            ColumnSection().update(r, r.compute_whitespaces(min_width, x_min, x_max)) for r in rows
        ]

    # Compute median row separation in core rows
    row_gaps = [
        nxt.y_center - prv.y_center
        for prv, nxt in pairwise(rows[min(core_indices) : max(core_indices) + 1])
    ]
    threshold = 1.5 * np.median(row_gaps)

    # Identify if top elements are coherent with core rows
    top_split_idx: int | None = None
    for i in range(min(core_indices) - 1, -1, -1):
        if abs(rows[i + 1].y_center - rows[i].y_center) > threshold:
            top_split_idx = i
            break

    # Identify if bottom elements are coherent with core rows
    bottom_split_idx: int | None = None
    for i in range(max(core_indices) + 1, len(rows)):
        if abs(rows[i].y_center - rows[i - 1].y_center) > threshold:
            bottom_split_idx = i
            break

    # Identify core section delimiters
    core_start = top_split_idx + 1 if top_split_idx is not None else 0
    core_end = bottom_split_idx if bottom_split_idx is not None else len(rows)

    # Create core rows section
    core_rows = rows[core_start:core_end]
    core_section = ColumnSection(
        items=[it for r in core_rows for it in r.items],
        rows=core_rows,
        whitespaces=compute_whitespaces(
            [it for r in core_rows for it in r.items], 0.5 * min_width, x_min, x_max
        ),
    )

    # Create top and bottom sections
    top_sections = [
        ColumnSection().update(row, row.compute_whitespaces(min_width, x_min, x_max))
        for row in rows[:core_start]
    ]
    bottom_sections = [
        ColumnSection().update(row, row.compute_whitespaces(min_width, x_min, x_max))
        for row in rows[core_end:]
    ]

    return [*top_sections, core_section, *bottom_sections]


def assess_columns_relevance(section: ColumnSection, min_col_width: float) -> ColumnSection:
    """
    Assess that all detected columns within a section are of a relevant size
    :param section: the section to assess
    :param min_col_width: the minimum column width
    :return: the section with irrelevant columns removed
    """
    # Identify irrelevant columns / merged whitespaces
    merged_whitespaces = []
    current_merge = None
    for prv, nxt in pairwise(section.whitespaces):
        if nxt.start - prv.end < min_col_width:
            if current_merge is None:
                current_merge = Whitespace(
                    start=prv.start,
                    end=nxt.end,
                    start_bound=prv.start_bound,
                    end_bound=nxt.end_bound,
                )
            else:
                current_merge.end = nxt.end
                current_merge.end_bound = nxt.end_bound
        elif current_merge is not None:
            merged_whitespaces.append(current_merge)
            current_merge = None

    if current_merge is not None:
        merged_whitespaces.append(current_merge)

    if len(merged_whitespaces) == 0:
        return section

    # Create new section with updated elements
    kept_items = [
        it
        for it in section.items
        if not any(ws.start <= it.x1 and ws.end >= it.x2 for ws in merged_whitespaces)
    ]
    kept_whitespaces = [
        ws
        for ws in section.whitespaces
        if not any(m_ws.start <= ws.start and m_ws.end >= ws.end for m_ws in merged_whitespaces)
    ]

    return ColumnSection(
        items=kept_items,
        rows=identify_merged_rows(cnts=kept_items),
        whitespaces=sorted([*merged_whitespaces, *kept_whitespaces], key=lambda ws: ws.start),
    )


def ensure_consistent_section(
    section: ColumnSection, min_width: float, x_min: int, x_max: int
) -> list[ColumnSection]:
    """
    Ensure consistency of section elements and columns
    :param section: Column section to check.
    :param min_width: minimum width for a whitespace to be considered
    :param x_min: start of span
    :param x_max: end of span
    :return: List of consistent column sections.
    """
    # Check that top / bottom elements of the column section are consistent with the rest of the section
    splitted_sections = ensure_section_bounds_consistency(
        section=section, min_width=min_width, x_min=x_min, x_max=x_max
    )

    # Pass over each section to check on column consistency
    return [
        assess_columns_relevance(section=sec, min_col_width=2 * min_width)
        for sec in splitted_sections
    ]
