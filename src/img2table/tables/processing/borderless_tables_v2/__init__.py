import numpy as np

from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables_v2._model import ColumnSection
from img2table.tables.processing.borderless_tables_v2.layout import identify_image_layout
from img2table.tables.processing.borderless_tables_v2.sections import identify_column_sections


def extract_borderless_tables(
    thresh: np.ndarray,
    lines: list[Line],
    char_length: float,
    existing_tables: list[Table] | None = None,
) -> list[ColumnSection]:
    """
    Identify borderless tables in a thresholded image.
    :param thresh: Thresholded image.
    :param lines: List of detected lines.
    :param char_length: Character length in pixels.
    :param existing_tables: Existing bordered tables.
    """
    # Identify layout from the thresholded image
    layout_regions = identify_image_layout(
        thresh=thresh, lines=lines, char_length=char_length, existing_tables=existing_tables
    )

    column_sections = []
    for region in layout_regions:
        # Get column sections
        column_sections += identify_column_sections(layout_region=region, min_width=char_length)

    return column_sections
