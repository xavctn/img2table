import numpy as np

from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables_v2.sections import compute_column_sections
from img2table.tables.processing.borderless_tables_v2.text_lines import identify_image_contours


def extract_borderless_tables(
    thresh: np.ndarray,
    lines: list[Line],
    char_length: float,
    existing_tables: list[Table] | None = None,
) -> None:
    """
    Identify borderless tables in a thresholded image.
    :param thresh: Thresholded image.
    :param lines: List of detected lines.
    :param char_length: Character length in pixels.
    :param existing_tables: Existing bordered tables.
    """
    # Identify contours in the thresholded image
    contours = identify_image_contours(
        thresh=thresh, lines=lines, char_length=char_length, existing_tables=existing_tables
    )

    # Get column sections
    column_sections = compute_column_sections(
        contours=contours, min_width=char_length, width=thresh.shape[1]
    )
