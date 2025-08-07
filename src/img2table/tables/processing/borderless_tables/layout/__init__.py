from typing import Optional

import numpy as np

from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.layout.column_segments import segment_image_columns
from img2table.tables.processing.borderless_tables.layout.image_elements import get_image_elements
from img2table.tables.processing.borderless_tables.layout.rlsa import identify_text_mask
from img2table.tables.processing.borderless_tables.layout.table_segments import get_table_segments
from img2table.tables.processing.borderless_tables.model import TableSegment, ImageSegment


def segment_image(thresh: np.ndarray, lines: list[Line], char_length: float,
                  median_line_sep: float, existing_tables: Optional[list[Table]] = None) -> list[TableSegment]:
    """
    Segment image and its elements
    :param thresh: threshold image array
    :param lines: list of Line objects of the image
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param existing_tables: list of detected bordered tables
    :return: list of ImageSegment objects with corresponding elements
    """
    # Identify text mask
    text_thresh = identify_text_mask(thresh=thresh,
                                     lines=lines,
                                     char_length=char_length,
                                     existing_tables=existing_tables)

    # Identify image elements
    img_elements = get_image_elements(thresh=text_thresh,
                                      char_length=char_length)

    if len(img_elements) == 0:
        return []

    # Identify column segments
    y_min, y_max = min([el.y1 for el in img_elements]), max([el.y2 for el in img_elements])
    image_segment = ImageSegment(x1=0, y1=y_min, x2=thresh.shape[1], y2=y_max, elements=img_elements)

    col_segments = segment_image_columns(image_segment=image_segment,
                                         char_length=char_length,
                                         lines=lines)

    # Within each column, identify segments that can correspond to tables
    return [table_segment for col_segment in col_segments
            for table_segment in get_table_segments(segment=col_segment,
                                                    char_length=char_length,
                                                    median_line_sep=median_line_sep)
            ]
