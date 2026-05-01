from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.tables.borderless.tables.structure.convert import (
    section_group_to_table,
)
from img2table.tables.borderless.tables.structure.discrepency import (
    bridge_small_discrepencies,
)

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
    # Create groups of sections that are likely to be part of the same table
    table_groups = bridge_small_discrepencies(
        column_sections=column_sections,
        char_length=char_length,
        height=height,
        width=width,
    )

    # Create tables from table groups
    return [section_group_to_table(gp) for gp in table_groups]
