from itertools import pairwise

from img2table.tables import find_components
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables_v2._model import ColumnSection
from img2table.tables.processing.borderless_tables_v2.tables.filter.model import StructuredSection


# TODO: Handle varying number of columns -> creating merged rows or discarding separators in non structured data
def _section_group_to_table(section_group: list[StructuredSection]) -> Table:
    """
    Create table from sections
    :param column_sections: list of structued sections
    :return: created table
    """
    # Compute vertical separators
    separators = [
        min(sec.y_min for sec in section_group),
        *[(prv.y_max + nxt.y_min) // 2 for prv, nxt in pairwise(section_group)],
        max(sec.y_max for sec in section_group),
    ]

    # Compute all tables
    sections_tbs = [
        sec.table(
            x_min=min(sec.x_min for sec in section_group),
            x_max=max(sec.x_max for sec in section_group),
            y_min=y_min,
            y_max=y_max,
        )
        for sec, (y_min, y_max) in zip(section_group, pairwise(separators), strict=True)
    ]

    # Create resulting table
    return Table(rows=[row for sec_tb in sections_tbs for row in sec_tb.items])


def bridge_small_discrepencies(
    column_sections: list[ColumnSection], width: int, height: int, char_length: float
) -> list[Table]:
    """
    Create groups of tables by bridging small discrepencies
    :param column_sections: list of column sections
    :param width: full page width in pixels
    :param height: full page height in pixels
    :param char_length: Character length in pixels.
    :return: list of created tables.
    """
    column_sections = sorted(column_sections, key=lambda sec: sec.y1)

    # Map to structured sections
    structured_sections = [
        StructuredSection.from_section(
            section=section, width=width, height=height, char_length=char_length
        )
        for section in column_sections
    ]

    # Identify groups of consecutive sections that have small enough discrepencies
    edges = [{idx} for idx, struct in enumerate(structured_sections) if struct.is_structured()]

    for idx, (prv, nxt) in enumerate(pairwise(structured_sections)):
        if min(prv.nb_columns, nxt.nb_columns) < 2:
            # Too few columns
            continue
        if max(prv.nb_columns, nxt.nb_columns) - min(prv.nb_columns, nxt.nb_columns) > 1:
            # Discrepency in column numbers
            continue
        if not max(prv.is_structured(), nxt.is_structured()):
            # Not any structured section
            continue
        if not all(sec.is_structured() or len(sec.merged_rows) < 3 for sec in [prv, nxt]):
            # Check that all elements are structured or have less than 3 rows
            continue

        # Compute distance between sections
        distance = nxt.y_min - prv.y_max
        if distance >= max(prv.row_height, nxt.row_height):
            # Too far away from each other
            continue

        # Check correspondance between whitespaces
        ws_small, ws_large = (
            (prv.whitespaces, nxt.whitespaces)
            if prv.nb_columns < nxt.nb_columns
            else (nxt.whitespaces, prv.whitespaces)
        )
        if sum(
            1
            for ws_l in ws_large
            if any(ws for ws in ws_small if min(ws_l.end, ws.end) - max(ws_l.start, ws.start) > 0)
        ) == max(prv.nb_columns, nxt.nb_columns):
            edges.append({idx, idx + 1})

    # Identify groups of related sections that form a table
    table_groups = [
        [structured_sections[idx] for idx in sorted(cluster)]
        for cluster in find_components(edges=edges)
    ]

    return [_section_group_to_table(section_group=group) for group in table_groups]
