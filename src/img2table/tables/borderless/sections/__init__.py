from img2table.tables.borderless.sections.consistency import (
    ensure_consistent_section,
)
from img2table.tables.borderless.sections.merging import (
    merge_column_sections,
)
from img2table.tables.borderless.sections.segmentation import (
    compute_column_sections,
)
from img2table.tables.borderless.types import (
    ColumnSection,
    LayoutRegion,
    identify_merged_rows,
)


def identify_column_sections(layout_region: LayoutRegion, min_width: float) -> list[ColumnSection]:
    """
    Identify column sections from a list of contours of the region.
    :param layout_region: region of the image containing contours.
    :param min_width: Minimum width of a column delimiter.
    :return: List of column sections.
    """
    # Compute merged rows
    merged_rows = identify_merged_rows(cnts=layout_region.contours)

    # Identify column sections
    column_sections, max_gap = compute_column_sections(
        merged_rows=merged_rows,
        min_width=min_width,
        x_min=layout_region.x1,
        x_max=layout_region.x2,
        ratio_vertical_separation=3,
    )

    # Merge column sections
    merged_sections = merge_column_sections(
        column_sections=column_sections,
        min_width=min_width,
        max_gap=max_gap,
        x_min=layout_region.x1,
        x_max=layout_region.x2,
    )

    # Check consistency and relevance of section elements
    return [
        sec
        for section in merged_sections
        for sec in ensure_consistent_section(
            section=section,
            min_width=min_width,
            x_min=layout_region.x1,
            x_max=layout_region.x2,
        )
    ]
