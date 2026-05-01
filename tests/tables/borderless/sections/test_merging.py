import json
from pathlib import Path

from img2table.tables.borderless.sections.merging import merge_column_sections
from img2table.tables.borderless.sections.segmentation import (
    compute_column_sections,
)
from img2table.tables.borderless.types import identify_merged_rows
from img2table.tables.types import Cell


def test_merge_column_sections() -> None:
    with Path("test_data/contours.json").open() as f:
        contours = [Cell(**el) for el in json.load(f)]
    merged_rows = identify_merged_rows(cnts=contours)

    sections, max_gap = compute_column_sections(
        merged_rows=merged_rows, min_width=9.0, x_min=0, x_max=2339, ratio_vertical_separation=3
    )

    merged_sections = merge_column_sections(
        column_sections=sections, min_width=9.0, x_min=0, x_max=2339, max_gap=max_gap
    )

    assert len(merged_sections) == 1
    assert [sec.nb_columns for sec in merged_sections] == [10]
