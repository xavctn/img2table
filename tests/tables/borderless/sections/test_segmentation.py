import json
from pathlib import Path

from img2table.tables.borderless.sections.segmentation import (
    compute_column_sections,
)
from img2table.tables.borderless.types import identify_merged_rows
from img2table.tables.types import Cell


def test_compute_column_section() -> None:
    with Path("test_data/contours.json").open() as f:
        contours = [Cell(**el) for el in json.load(f)]
    merged_rows = identify_merged_rows(cnts=contours)

    sections, max_gap = compute_column_sections(
        merged_rows=merged_rows, min_width=9.0, x_min=0, x_max=2339, ratio_vertical_separation=3
    )

    assert max_gap == 117.75
    assert len(sections) == 7
    assert [sec.nb_columns for sec in sections] == [11, 13, 10, 11, 10, 11, 12]
