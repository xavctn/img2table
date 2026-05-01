from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

from img2table.tables.borderless.types import Whitespace
from img2table.tables.common import cluster_items
from img2table.tables.types import Table

if TYPE_CHECKING:
    from img2table.tables.borderless.tables.filter.model import (
        StructuredSection,
    )


def _reference_column_separators(section_group: list[StructuredSection]) -> list[Whitespace]:
    """
    Compute the reference column separators for the section group.
    :param section_group: list of structured sections
    :return: list of reference column separators
    """
    # Keep only structured sections for column separators references
    ref_sections = [sec for sec in section_group if sec.is_structured()]

    if len(ref_sections) == 0:
        return []
    if len(ref_sections) == len(section_group) == 1:
        return ref_sections[0].whitespaces
    if len(ref_sections) == 1:
        # Set reference whitespaces as the unique reference section whitespaces
        ref_whitespaces = ref_sections[0].whitespaces
    else:
        # Get unit whitespaces for structured sections and cluster them
        structured_unit_ws = [
            ws
            for sec in ref_sections
            for ws in sec.whitespaces
            if max(
                sum(ws.overlaps(other_ws) for other_ws in other_sec.whitespaces)
                for other_sec in ref_sections
                if sec != other_sec
            )
            == 1
        ]
        ws_clusters = cluster_items(
            items=structured_unit_ws,
            clustering_func=lambda ws1, ws2: ws1.overlaps(ws2),
        )

        # Get whitespaces from cluster
        ref_whitespaces = [
            Whitespace(
                start=min(max(ws.start for ws in cl), *(ws.end for ws in cl)),
                end=max(*(ws.start for ws in cl), min(ws.end for ws in cl)),
                start_bound=min(ws.start_bound for ws in cl),
                end_bound=min(ws.end_bound for ws in cl),
            )
            for cl in ws_clusters
        ]

    # Try to reduce reference whitespaces based on other "non structured" sections
    unit_other_ws = [
        ws
        for sec in section_group
        if not sec.is_structured()
        for ws in sec.whitespaces
        if sum(ws.overlaps(ref_ws) for ref_ws in ref_whitespaces) == 1
    ]

    final_whitespaces: list[Whitespace] = []
    for ref_ws in ref_whitespaces:
        # Get matching whitespaces from other sections
        matching_ws = [other_ws for other_ws in unit_other_ws if other_ws.overlaps(ref_ws)]
        final_whitespaces.append(
            Whitespace(
                start=max((ref_ws.start, *(ws.start for ws in matching_ws))),
                end=min((ref_ws.end, *(ws.end for ws in matching_ws))),
            )
        )

    return sorted(final_whitespaces, key=lambda ws: ws.start)


def section_group_to_table(section_group: list[StructuredSection]) -> Table:
    """
    Create a table from aligned sections while handling one-column discrepancies.
    :param section_group: list of structured sections
    :return: created table
    """
    section_group = sorted(section_group, key=lambda sec: sec.y_min)

    if len(section_group) == 0:
        raise ValueError("Empty group")
    if len(section_group) == 1:
        return next(iter(section_group)).table()

    # Compute vertical separators between section elemnts
    separators = [
        min(sec.y_min for sec in section_group),
        *[(prv.y_max + nxt.y_min) // 2 for prv, nxt in pairwise(section_group)],
        max(sec.y_max for sec in section_group),
    ]

    # Compute reference whitespaces
    ref_ws = _reference_column_separators(section_group=section_group)

    # Compute all tables
    sections_tbs = [
        sec.table(
            x_min=min(sec.x_min for sec in section_group),
            x_max=max(sec.x_max for sec in section_group),
            y_min=y_min,
            y_max=y_max,
            ref_whitespaces=ref_ws,
        )
        for sec, (y_min, y_max) in zip(section_group, pairwise(separators), strict=True)
    ]

    # Create resulting table
    return Table(rows=[row for sec_tb in sections_tbs for row in sec_tb.rows])
