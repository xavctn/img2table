# coding: utf-8
from typing import List

from img2table.tables.processing.borderless_tables.column_delimiters.column_groups import create_delimiter_groups
from img2table.tables.processing.borderless_tables.column_delimiters.vertical_whitespaces import \
    get_relevant_vertical_whitespaces
from img2table.tables.processing.borderless_tables.model import ImageSegment, DelimiterGroup


def identify_column_groups(segment: ImageSegment, char_length: float) -> List[DelimiterGroup]:
    """
    Identify list of vertical delimiters that can be table columns in an image segment
    :param segment: image segment
    :param char_length: average character width in image
    :return: list of delimiter groups that can correspond to columns
    """
    # Identify vertical whitespaces in segment
    vertical_ws = get_relevant_vertical_whitespaces(segment=segment,
                                                    char_length=char_length)

    # Get delimiter groups that can correspond to columns
    delimiter_groups = create_delimiter_groups(delimiters=vertical_ws,
                                               segment=segment)

    return [gp for gp in delimiter_groups if len(gp.delimiters) >= 4]
