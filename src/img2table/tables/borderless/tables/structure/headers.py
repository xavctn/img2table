from __future__ import annotations

from itertools import pairwise

from img2table.tables.borderless.tables.filter.model import StructuredSection
from img2table.tables.borderless.tables.structure.convert import _reference_column_separators
from img2table.tables.borderless.types import Whitespace


def header_ws_matches(header_ws: list[Whitespace], ref_ws: list[Whitespace]) -> bool:
    """
    Identify if header whitespaces correspond to reference whitespaces
    :param header_ws: header whitespaces
    :param ref_ws: reference whitespaces.
    :return: boolean indicating if header whitespaces correspond to reference whitespaces
    """
    # Check cover of header whitespaces columns versus reference columns
    cover_map, covered_idx = {}, set()
    for col_idx_header, (prv_h, nxt_h) in enumerate(pairwise(header_ws)):
        # Get matching reference columns
        matching_columns = {
            col_idx_ref
            for col_idx_ref, (prv_r, nxt_r) in enumerate(pairwise(ref_ws))
            if min(nxt_r.start, nxt_h.start) - max(prv_r.end, prv_h.end) > 0
        }

        # Check that no header columns are covering the same column
        if matching_columns.intersection(covered_idx):
            return False

        cover_map[col_idx_header] = matching_columns
        covered_idx |= matching_columns

    return (
        len(cover_lengths := set(map(len, cover_map.values()))) == 1
        and next(iter(cover_lengths)) >= 2
    )


def update_header_whitespaces(
    header_section: StructuredSection, ref_ws: list[Whitespace]
) -> StructuredSection:
    """
    Update header whitespaces based on reference whitespaces
    :param header_section: header section
    :param ref_ws: reference whitespaces.
    :return: section with updated whitespaces
    """
    # Replace left and right whitespaces
    left_ws = [
        Whitespace(start=ws.start, end=min(ws.end, header_section.x_min))
        for ws in ref_ws
        if ws.start < header_section.x_min
    ] or header_section.whitespaces[:1]
    right_ws = [
        Whitespace(start=max(ws.start, header_section.x_max), end=ws.end)
        for ws in ref_ws
        if ws.end > header_section.x_max
    ] or header_section.whitespaces[-1:]
    updated_ws = [*left_ws, *header_section.whitespaces[1:-1], *right_ws]

    return StructuredSection(
        idx=header_section.idx,
        height=header_section.height,
        width=header_section.width,
        char_length=header_section.char_length,
        items=header_section.items,
        merged_rows=header_section.merged_rows,
        whitespaces=updated_ws,
    )


def identify_potentials_headers(
    structured_sections: list[StructuredSection], table_groups: list[list[StructuredSection]]
) -> list[list[StructuredSection]]:
    """
    Identify if some sections can be used as headers of previously identified table table_groups
    :param structured_sections: list of all structured sections
    :param table_groups: list of clusters of sections that correspond to the same table.
    :return: updated clusters with headers
    """
    # Identify unused sections
    unused_sections = {
        section.idx: section
        for section in structured_sections
        if section.idx not in {sec.idx for group in table_groups for sec in group}
    }

    final_groups = []
    for group in table_groups:
        # Identify header
        header_idx = min(sec.idx for sec in group) - 1
        if header_idx not in unused_sections:
            final_groups.append(group)
            continue
        header_section = unused_sections[header_idx]

        # Identify is the section might correspond with the table group
        distance = min(sec.y_min for sec in group) - header_section.y_max
        if (
            distance >= 3 * max(sec.row_height for sec in group)
            or header_section.nb_columns < 2
            or header_section.nb_columns
            >= min(sec.nb_columns for sec in group if sec.is_structured())
        ):
            final_groups.append(group)
            continue

        # Compute column separators from group
        group_ws = _reference_column_separators(section_group=group)
        if header_ws_matches(header_ws=header_section.whitespaces, ref_ws=group_ws):
            final_groups.append([update_header_whitespaces(header_section, group_ws), *group])
        else:
            final_groups.append(group)

    return final_groups
