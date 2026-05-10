from img2table.tables.borderless.tables.filter.metrics import (
    TableMetrics,
)
from img2table.tables.borderless.tables.filter.model import (
    StructuredSection,
)
from img2table.tables.borderless.types import (
    ColumnSection,
    MergedRow,
    Whitespace,
)
from img2table.tables.types import Cell, Row, Table

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


whitespaces = [
    Whitespace(start=0, end=10, start_bound=True),
    Whitespace(start=20, end=40),
    Whitespace(start=50, end=70),
    Whitespace(start=80, end=100, end_bound=True),
]


section = StructuredSection(
    height=100,
    width=100,
    char_length=5,
    items=[item for row in rows for item in row.items],
    merged_rows=rows,
    whitespaces=whitespaces,
    _row_ranges=[(10, 25), (25, 45), (45, 60)],
)


def test_from_section() -> None:
    section = ColumnSection(
        items=[item for row in rows for item in row.items],
        rows=rows,
        whitespaces=whitespaces,
    )

    result = StructuredSection.from_section(idx=0, section=section, width=120, height=80, char_length=6)

    assert result == StructuredSection(
        height=80,
        width=120,
        char_length=6,
        items=section.items,
        merged_rows=section.rows,
        whitespaces=section.whitespaces,
    )


def test_table_score(
    monkeypatch,  # noqa: ANN001
) -> None:
    sec = StructuredSection(
        height=100,
        width=100,
        char_length=5,
        items=section.items[:6],
        merged_rows=rows[:2],
        whitespaces=whitespaces,
        _row_ranges=[(10, 25), (25, 45)],
    )

    monkeypatch.setattr(
        TableMetrics,
        "from_section",
        classmethod(
            lambda cls, section: cls(  # noqa: ARG005
                presence_ratios=[1.0, 1.0, 1.0],
                network_connectivity=1.0,
                mean_column_alignment=1.0,
                min_column_alignment=1.0,
                spacing_consistency=1.0,
                row_pattern_consistency=1.0,
                sparsity=1.0,
                full_text=0.0,
            )
        ),
    )

    assert sec.table_score == 0.0
    assert not sec.is_structured()


def test_table() -> None:
    result = section.table(
        x_min=5,
        x_max=95,
        y_min=8,
        y_max=62,
        ref_whitespaces=[
            Whitespace(start=0, end=12, start_bound=True),
            Whitespace(start=24, end=36),
            Whitespace(start=58, end=72),
            Whitespace(start=88, end=100, end_bound=True),
        ],
    )

    assert result == Table(
        rows=[
            Row(
                cells=[
                    Cell(x1=5, y1=8, x2=30, y2=25),
                    Cell(x1=30, y1=8, x2=64, y2=25),
                    Cell(x1=64, y1=8, x2=95, y2=25),
                ]
            ),
            Row(
                cells=[
                    Cell(x1=5, y1=25, x2=30, y2=45),
                    Cell(x1=30, y1=25, x2=64, y2=45),
                    Cell(x1=64, y1=25, x2=95, y2=45),
                ]
            ),
            Row(
                cells=[
                    Cell(x1=5, y1=45, x2=30, y2=62),
                    Cell(x1=30, y1=45, x2=64, y2=62),
                    Cell(x1=64, y1=45, x2=95, y2=62),
                ]
            ),
        ]
    )
