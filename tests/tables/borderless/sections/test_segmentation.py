import json
from pathlib import Path

from img2table.tables.borderless.sections.segmentation import (
    compute_column_sections,
    matching_whitespaces,
)
from img2table.tables.borderless.types import Whitespace, identify_merged_rows
from img2table.tables.types import Cell


def test_matching_whitespaces() -> None:
    ws1 = [Whitespace(start=0, end=10, start_bound=True), Whitespace(start=20, end=50)]
    ws2 = [
        Whitespace(start=0, end=10, start_bound=True),
        Whitespace(start=20, end=30),
        Whitespace(start=35, end=50),
    ]

    is_match, matching_ws = matching_whitespaces(ws1_list=ws1, ws2_list=ws2, min_width=4)

    assert is_match
    assert matching_ws == ws2

    ws1 = [Whitespace(start=0, end=10, start_bound=True), Whitespace(start=20, end=30)]
    ws2 = [
        Whitespace(start=0, end=10, start_bound=True),
        Whitespace(start=20, end=30),
        Whitespace(start=40, end=50, end_bound=True),
    ]

    is_match, matching_ws = matching_whitespaces(ws1_list=ws1, ws2_list=ws2, min_width=4)

    assert not is_match
    assert matching_ws == []


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
