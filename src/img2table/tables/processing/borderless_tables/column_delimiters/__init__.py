# coding: utf-8
from typing import Optional

from img2table.tables.processing.borderless_tables.column_delimiters.columns import get_column_whitespaces
from img2table.tables.processing.borderless_tables.column_delimiters.vertical_whitespaces import \
    get_vertical_whitespaces
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableSegment


def identify_column_groups(table_segment: TableSegment, char_length: float,
                           median_line_sep: float) -> Optional[DelimiterGroup]:
    """
    Identify list of vertical delimiters that can be table columns in a table segment
    :param table_segment: table segment
    :param char_length: average character width in image
    :param median_line_sep: median line separation
    :return: delimiter group that can correspond to columns
    """
    # Identify vertical whitespaces in the table segment
    vertical_ws, unused_ws = get_vertical_whitespaces(table_segment=table_segment)

    if len(vertical_ws) == 0 or len(table_segment.elements) == 0:
        return None

    # Create delimiter group from whitespace
    delimiter_group = get_column_whitespaces(vertical_ws=vertical_ws,
                                             unused_ws=unused_ws,
                                             table_segment=table_segment,
                                             char_length=char_length,
                                             median_line_sep=median_line_sep)

    if len(delimiter_group.delimiters) >= 4 and len(delimiter_group.elements) > 0:
        return delimiter_group
    else:
        return None
