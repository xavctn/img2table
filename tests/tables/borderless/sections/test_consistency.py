from img2table.tables.borderless.sections.consistency import (
    assess_columns_relevance,
    ensure_consistent_section,
    ensure_section_bounds_consistency,
)
from img2table.tables.borderless.types import (
    ColumnSection,
    MergedRow,
    Whitespace,
)
from img2table.tables.types import Cell


def test_ensure_section_bounds_consistency_core() -> None:
    # Only core rows
    rows = [
        MergedRow(items=[Cell(x1=10, y1=0, x2=20, y2=10)]),
        MergedRow(
            items=[
                Cell(x1=10, y1=20, x2=20, y2=30),
                Cell(x1=40, y1=20, x2=50, y2=30),
                Cell(x1=80, y1=20, x2=90, y2=30),
            ]
        ),
        MergedRow(items=[Cell(x1=10, y1=40, x2=20, y2=50)]),
    ]

    result = ensure_section_bounds_consistency(
        section=ColumnSection(
            items=[item for row in rows for item in row.items],
            rows=rows,
            whitespaces=rows[1].compute_whitespaces(10, 0, 100),
        ),
        min_width=10,
        x_min=0,
        x_max=100,
    )

    assert len(result) == 3
    assert [len(sec.rows) for sec in result] == [1, 1, 1]
    assert [sec.rows[0] for sec in result] == rows


def test_ensure_section_bounds_consistency() -> None:
    top_row = MergedRow(items=[Cell(x1=10, y1=10, x2=20, y2=20)])
    core_rows = [
        MergedRow(
            items=[
                Cell(x1=10, y1=40, x2=20, y2=50),
                Cell(x1=40, y1=40, x2=50, y2=50),
                Cell(x1=80, y1=40, x2=90, y2=50),
            ]
        ),
        MergedRow(
            items=[
                Cell(x1=10, y1=50, x2=20, y2=60),
                Cell(x1=40, y1=50, x2=50, y2=60),
                Cell(x1=80, y1=50, x2=90, y2=60),
            ]
        ),
        MergedRow(
            items=[
                Cell(x1=10, y1=60, x2=20, y2=70),
                Cell(x1=40, y1=60, x2=50, y2=70),
                Cell(x1=80, y1=60, x2=90, y2=70),
            ]
        ),
    ]
    bottom_row = MergedRow(items=[Cell(x1=10, y1=110, x2=20, y2=120)])
    rows = [top_row, *core_rows, bottom_row]

    result = ensure_section_bounds_consistency(
        section=ColumnSection(
            items=[item for row in rows for item in row.items],
            rows=rows,
            whitespaces=core_rows[0].compute_whitespaces(10, 0, 100),
        ),
        min_width=10,
        x_min=0,
        x_max=100,
    )

    assert len(result) == 3
    assert [len(sec.rows) for sec in result] == [1, 3, 1]
    assert result[0].rows == [top_row]
    assert result[1].rows == core_rows
    assert result[2].rows == [bottom_row]


def test_assess_columns_relevance() -> None:
    kept_item = Cell(x1=35, y1=0, x2=65, y2=10)
    discarded_items = [Cell(x1=12, y1=0, x2=18, y2=10), Cell(x1=82, y1=0, x2=88, y2=10)]
    section = ColumnSection(
        items=[discarded_items[0], kept_item, discarded_items[1]],
        rows=[MergedRow(items=[discarded_items[0], kept_item, discarded_items[1]])],
        whitespaces=[
            Whitespace(start=0, end=10, start_bound=True),
            Whitespace(start=20, end=30),
            Whitespace(start=70, end=80),
            Whitespace(start=90, end=100, end_bound=True),
        ],
    )

    result = assess_columns_relevance(section=section, min_col_width=20)

    assert result.items == [kept_item]
    assert len(result.rows) == 1
    assert [(ws.start, ws.end) for ws in result.whitespaces] == [(0, 30), (70, 100)]


def test_ensure_consistent_section() -> None:
    top_row = MergedRow(items=[Cell(x1=10, y1=10, x2=20, y2=20)])
    core_rows = [
        MergedRow(
            items=[
                Cell(x1=10, y1=40, x2=20, y2=50),
                Cell(x1=40, y1=40, x2=50, y2=50),
                Cell(x1=80, y1=40, x2=90, y2=50),
            ]
        ),
        MergedRow(
            items=[
                Cell(x1=10, y1=50, x2=20, y2=60),
                Cell(x1=40, y1=50, x2=50, y2=60),
                Cell(x1=80, y1=50, x2=90, y2=60),
            ]
        ),
        MergedRow(
            items=[
                Cell(x1=10, y1=60, x2=20, y2=70),
                Cell(x1=40, y1=60, x2=50, y2=70),
                Cell(x1=80, y1=60, x2=90, y2=70),
            ]
        ),
    ]
    bottom_row = MergedRow(items=[Cell(x1=10, y1=110, x2=20, y2=120)])
    rows = [top_row, *core_rows, bottom_row]
    section = ColumnSection(
        items=[item for row in rows for item in row.items],
        rows=rows,
        whitespaces=[
            Whitespace(start=0, end=10, start_bound=True),
            Whitespace(start=20, end=25),
            Whitespace(start=35, end=45),
            Whitespace(start=55, end=60),
            Whitespace(start=90, end=100, end_bound=True),
        ],
    )

    result = ensure_consistent_section(section=section, min_width=10, x_min=0, x_max=100)

    assert len(result) == 3
    assert [len(sec.rows) for sec in result] == [0, 0, 0]
    assert [(ws.start, ws.end) for ws in result[1].whitespaces] == [(0, 100)]
