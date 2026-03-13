from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables_v2._model import ColumnSection
from img2table.tables.processing.borderless_tables_v2.sections.merging import (
    ensure_section_bounds_consistency,
    merge_column_sections,
)
from img2table.tables.processing.borderless_tables_v2.sections.segmentation import (
    compute_column_section,
    identify_merged_rows,
)


def compute_column_sections(
    contours: list[Cell], min_width: float, width: int
) -> list[ColumnSection]:
    """
    Identify column sections from a list of contours.
    :param contours: List of contours to identify column sections from.
    :param min_width: Minimum width of a column delimiter.
    :param width: Width of the image.
    :return: List of column sections.
    """
    # Compute merged rows
    merged_rows = identify_merged_rows(cnts=contours)

    # Identify column sections
    column_sections, max_gap = compute_column_section(
        merged_rows=merged_rows, min_width=min_width, width=width, ratio_vertical_separation=3
    )

    # Merge column sections
    merged_sections = merge_column_sections(
        column_sections=column_sections, min_width=min_width, max_gap=max_gap, width=width
    )

    # Check sections top and bottom bounds
    return [
        sec
        for section in merged_sections
        for sec in ensure_section_bounds_consistency(
            section=section, min_width=min_width, width=width
        )
    ]
