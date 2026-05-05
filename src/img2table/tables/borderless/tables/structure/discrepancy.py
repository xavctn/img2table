from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

from img2table.tables.common import find_components

if TYPE_CHECKING:
    from img2table.tables.borderless.tables.filter.model import StructuredSection


def bridge_small_discrepancies(
    structured_sections: list[StructuredSection],
) -> list[list[StructuredSection]]:
    """
    Create groups of tables by bridging small discrepancies
    :param structured_sections: list of structured column sections
    :return: list of clusters of sections that correspond to the same table.
    """
    # Identify groups of consecutive sections that have small enough discrepancies
    structured_sections = sorted(structured_sections, key=lambda sec: sec.idx)
    edges = [{idx} for idx, struct in enumerate(structured_sections) if struct.is_structured()]

    for idx, (prv, nxt) in enumerate(pairwise(structured_sections)):
        if min(prv.nb_columns, nxt.nb_columns) < 3:
            # Too few columns
            continue
        if max(prv.nb_columns, nxt.nb_columns) - min(prv.nb_columns, nxt.nb_columns) > 1:
            # Discrepancy in column numbers
            continue
        if not max(prv.is_structured(), nxt.is_structured()):
            # Not any structured section
            continue
        if not all(
            sec.is_structured() or sec.nb_rows < 2 or len(sec.merged_rows) < 3 for sec in [prv, nxt]
        ):
            # Check that all elements are structured or have less than 3 rows
            continue

        # Compute distance between sections
        distance = nxt.y_min - prv.y_max
        if distance >= max(prv.row_height, nxt.row_height):
            # Too far away from each other
            continue

        # Check correspondence between whitespaces
        ws_small, ws_large = (
            (prv.whitespaces, nxt.whitespaces)
            if prv.nb_columns < nxt.nb_columns
            else (nxt.whitespaces, prv.whitespaces)
        )

        # If all whitespaces match (except one), the sections are likely connected
        nb_matching_whitespaces = sum(
            1
            for ws_l in ws_large
            if any(ws for ws in ws_small if min(ws_l.end, ws.end) - max(ws_l.start, ws.start) > 0)
        )
        if nb_matching_whitespaces == max(prv.nb_columns, nxt.nb_columns):
            edges.append({idx, idx + 1})
        if nb_matching_whitespaces == len(prv.whitespaces) == len(nxt.whitespaces):
            edges.append({idx, idx + 1})

    # Identify groups of related sections that form a table
    return [
        [structured_sections[idx] for idx in sorted(cluster)]
        for cluster in find_components(edges=edges)
    ]
