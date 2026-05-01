import pytest

from img2table.tables.borderless.tables.filter.metrics import (
    TableMetrics,
)
from img2table.tables.borderless.tables.filter.metrics.columns import (
    compute_columns_metrics,
)
from img2table.tables.borderless.tables.filter.metrics.content import (
    column_presence_ratios,
    compute_content_layout_metrics,
    network_connectivity_score,
    row_pattern_consistency_score,
)
from img2table.tables.borderless.tables.filter.metrics.misc import (
    full_text_score,
    sparsity_score,
)
from img2table.tables.borderless.tables.filter.metrics.rows import (
    content_spacing_consistency,
)
from img2table.tables.borderless.tables.filter.model import (
    StructuredSection,
)
from img2table.tables.borderless.types import MergedRow, Whitespace
from img2table.tables.types import Cell


def _three_column_whitespaces() -> list[Whitespace]:
    return [
        Whitespace(start=0, end=10, start_bound=True),
        Whitespace(start=20, end=40),
        Whitespace(start=50, end=70),
        Whitespace(start=80, end=100, end_bound=True),
    ]


def _aligned_section() -> StructuredSection:
    rows = [
        MergedRow(
            items=[
                Cell(x1=10, y1=10, x2=20, y2=20),
                Cell(x1=40, y1=10, x2=50, y2=20),
                Cell(x1=70, y1=10, x2=80, y2=20),
            ]
        ),
        MergedRow(
            items=[
                Cell(x1=10, y1=30, x2=20, y2=40),
                Cell(x1=40, y1=30, x2=50, y2=40),
                Cell(x1=70, y1=30, x2=80, y2=40),
            ]
        ),
        MergedRow(
            items=[
                Cell(x1=10, y1=50, x2=20, y2=60),
                Cell(x1=40, y1=50, x2=50, y2=60),
                Cell(x1=70, y1=50, x2=80, y2=60),
            ]
        ),
    ]
    return StructuredSection(
        height=100,
        width=100,
        char_length=5,
        items=[item for row in rows for item in row.items],
        merged_rows=rows,
        whitespaces=_three_column_whitespaces(),
        _row_ranges=[(10, 25), (25, 45), (45, 60)],
    )


def test_compute_columns_metrics() -> None:
    result = compute_columns_metrics(section=_aligned_section())

    assert result == (1.0, 1.0)


def test_content_spacing_consistency() -> None:
    section = StructuredSection(
        height=100,
        width=100,
        char_length=5,
        items=[
            Cell(x1=10, y1=10, x2=20, y2=20),
            Cell(x1=40, y1=10, x2=50, y2=20),
            Cell(x1=70, y1=10, x2=80, y2=20),
            Cell(x1=10, y1=25, x2=20, y2=35),
            Cell(x1=40, y1=25, x2=50, y2=35),
            Cell(x1=70, y1=25, x2=80, y2=35),
            Cell(x1=10, y1=40, x2=20, y2=50),
            Cell(x1=40, y1=40, x2=50, y2=50),
            Cell(x1=70, y1=40, x2=80, y2=50),
        ],
        merged_rows=[
            MergedRow(
                items=[
                    Cell(x1=10, y1=10, x2=20, y2=20),
                    Cell(x1=40, y1=10, x2=50, y2=20),
                    Cell(x1=70, y1=10, x2=80, y2=20),
                ]
            ),
            MergedRow(
                items=[
                    Cell(x1=10, y1=25, x2=20, y2=35),
                    Cell(x1=40, y1=25, x2=50, y2=35),
                    Cell(x1=70, y1=25, x2=80, y2=35),
                ]
            ),
            MergedRow(
                items=[
                    Cell(x1=10, y1=40, x2=20, y2=50),
                    Cell(x1=40, y1=40, x2=50, y2=50),
                    Cell(x1=70, y1=40, x2=80, y2=50),
                ]
            ),
        ],
        whitespaces=_three_column_whitespaces(),
        _row_ranges=[(10, 25), (25, 40), (40, 55)],
    )

    assert content_spacing_consistency(section=section) == 1.0


def test_compute_content_layout_metrics() -> None:
    section = _aligned_section()

    result = compute_content_layout_metrics(section=section)

    assert result == ([1.0, 1.0, 1.0], 1.0, 1.0)


def test_content_helper_scores() -> None:
    section = _aligned_section()
    occupancy_matrix = [
        [True, True, False],
        [True, False, False],
        [False, False, True],
    ]

    assert column_presence_ratios(section=section, occupancy_matrix=occupancy_matrix) == [
        2 / 3,
        1 / 3,
        1 / 3,
    ]
    assert network_connectivity_score(section=section, occupancy_matrix=occupancy_matrix) == 0.25
    assert row_pattern_consistency_score(occupancy_matrix=occupancy_matrix) == 0.5


def test_full_text_score() -> None:
    section = StructuredSection(
        height=100,
        width=100,
        char_length=5,
        items=[
            Cell(x1=10, y1=10, x2=20, y2=20),
            Cell(x1=40, y1=10, x2=50, y2=20),
            Cell(x1=70, y1=10, x2=80, y2=20),
            Cell(x1=10, y1=30, x2=20, y2=40),
            Cell(x1=40, y1=30, x2=50, y2=40),
            Cell(x1=10, y1=50, x2=20, y2=60),
        ],
        merged_rows=[],
        whitespaces=_three_column_whitespaces(),
        _row_ranges=[(10, 20), (30, 40), (50, 60)],
    )

    assert full_text_score(section=section) == 1.0


def test_sparsity_score() -> None:
    section = StructuredSection(
        height=100,
        width=100,
        char_length=5,
        items=[
            Cell(x1=10, y1=10, x2=20, y2=20),
            Cell(x1=40, y1=10, x2=50, y2=20),
            Cell(x1=70, y1=10, x2=80, y2=20),
            Cell(x1=10, y1=30, x2=20, y2=40),
            Cell(x1=40, y1=30, x2=50, y2=40),
            Cell(x1=10, y1=50, x2=20, y2=60),
        ],
        merged_rows=[],
        whitespaces=_three_column_whitespaces(),
        _row_ranges=[(10, 20), (30, 40), (50, 60)],
    )

    assert sparsity_score(section=section) == pytest.approx(1 - abs((1 / 3) - 0.35) / 0.35)


def test_table_metrics_score() -> None:
    metrics = TableMetrics(
        presence_ratios=[1.0, 1.0, 1.0],
        network_connectivity=0.7,
        mean_column_alignment=0.8,
        min_column_alignment=0.6,
        spacing_consistency=0.9,
        row_pattern_consistency=0.5,
        sparsity=0.4,
        full_text=0.1,
    )

    assert metrics.score() == pytest.approx(0.5675)
