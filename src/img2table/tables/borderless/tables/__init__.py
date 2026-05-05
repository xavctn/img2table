from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.tables.borderless.tables.filter.model import StructuredSection
from img2table.tables.borderless.tables.structure.convert import (
    section_group_to_table,
)
from img2table.tables.borderless.tables.structure.discrepancy import (
    bridge_small_discrepancies,
)
from img2table.tables.borderless.tables.structure.headers import identify_potentials_headers

if TYPE_CHECKING:
    from img2table.tables.borderless.types import ColumnSection
    from img2table.tables.types import Table


def identify_tables(
    column_sections: list[ColumnSection], char_length: float, height: int, width: int
) -> list[Table]:
    """
    Identify tables in a list of column sections
    :param column_sections: list of column sections
    :param char_length: average character length
    :param height: image height
    :param width: image width
    :return: list of tables
    """
    # Map to structured sections
    structured_sections = [
        StructuredSection.from_section(
            idx=idx, section=section, width=width, height=height, char_length=char_length
        )
        for idx, section in enumerate(sorted(column_sections, key=lambda sec: sec.y1))
    ]

    # Create groups of sections that are likely to be part of the same table
    table_groups = bridge_small_discrepancies(
        structured_sections=structured_sections,
    )

    # Add potential headers
    table_groups_with_headers = identify_potentials_headers(
        structured_sections=structured_sections, table_groups=table_groups
    )

    # Create tables from table groups
    return [section_group_to_table(gp) for gp in table_groups_with_headers]
