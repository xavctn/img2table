import pytest

from img2table.tables.borderless.tables.filter.model import (
    StructuredSection,
)
from img2table.tables.borderless.tables.structure.convert import (
    _reference_column_separators,
    section_group_to_table,
)
from img2table.tables.borderless.types import MergedRow, Whitespace
from img2table.tables.types import Cell, Row, Table


def _structured_section(y_min: int, y_max: int, score: float = 0.5) -> StructuredSection:
    row = MergedRow(
        items=[
            Cell(x1=10, y1=y_min, x2=20, y2=y_min + 10),
            Cell(x1=40, y1=y_min, x2=50, y2=y_min + 10),
            Cell(x1=70, y1=y_min, x2=80, y2=y_min + 10),
        ]
    )
    section = StructuredSection(
        height=120,
        width=100,
        char_length=5,
        items=row.items,
        merged_rows=[row, MergedRow(items=[Cell(x1=10, y1=y_max - 10, x2=20, y2=y_max)])],
        whitespaces=[
            Whitespace(start=0, end=10, start_bound=True),
            Whitespace(start=20, end=40),
            Whitespace(start=50, end=70),
            Whitespace(start=80, end=100, end_bound=True),
        ],
        _row_ranges=[(y_min, (y_min + y_max) // 2), ((y_min + y_max) // 2, y_max)],
    )
    section.table_score = score  # type: ignore[attr-defined]
    return section


def test_reference_column_separators() -> None:
    structured = _structured_section(y_min=10, y_max=40)
    other = _structured_section(y_min=45, y_max=75, score=0.0)
    other.whitespaces = [
        Whitespace(start=0, end=11, start_bound=True),
        Whitespace(start=21, end=39),
        Whitespace(start=51, end=69),
        Whitespace(start=81, end=100, end_bound=True),
    ]

    result = _reference_column_separators(section_group=[structured, other])

    assert result == [
        Whitespace(start=0, end=10),
        Whitespace(start=21, end=39),
        Whitespace(start=51, end=69),
        Whitespace(start=81, end=100),
    ]


def test_section_group_to_table() -> None:
    top = StructuredSection(
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
        ],
        whitespaces=[
            Whitespace(start=0, end=10, start_bound=True),
            Whitespace(start=20, end=40),
            Whitespace(start=50, end=70),
            Whitespace(start=80, end=100, end_bound=True),
        ],
        _row_ranges=[(10, 22), (22, 35)],
    )
    top.table_score = 0.5  # type: ignore[attr-defined]
    bottom = StructuredSection(
        height=100,
        width=100,
        char_length=5,
        items=[
            Cell(x1=10, y1=40, x2=20, y2=50),
            Cell(x1=40, y1=40, x2=50, y2=50),
            Cell(x1=70, y1=40, x2=80, y2=50),
            Cell(x1=10, y1=55, x2=20, y2=65),
            Cell(x1=40, y1=55, x2=50, y2=65),
            Cell(x1=70, y1=55, x2=80, y2=65),
        ],
        merged_rows=[
            MergedRow(
                items=[
                    Cell(x1=10, y1=40, x2=20, y2=50),
                    Cell(x1=40, y1=40, x2=50, y2=50),
                    Cell(x1=70, y1=40, x2=80, y2=50),
                ]
            ),
            MergedRow(
                items=[
                    Cell(x1=10, y1=55, x2=20, y2=65),
                    Cell(x1=40, y1=55, x2=50, y2=65),
                    Cell(x1=70, y1=55, x2=80, y2=65),
                ]
            ),
        ],
        whitespaces=[
            Whitespace(start=0, end=10, start_bound=True),
            Whitespace(start=20, end=40),
            Whitespace(start=50, end=70),
            Whitespace(start=80, end=100, end_bound=True),
        ],
        _row_ranges=[(40, 52), (52, 65)],
    )
    bottom.table_score = 0.5  # type: ignore[attr-defined]

    result = section_group_to_table(section_group=[bottom, top])

    assert result == Table(
        rows=[
            Row(
                cells=[
                    Cell(x1=10, y1=10, x2=30, y2=22),
                    Cell(x1=30, y1=10, x2=60, y2=22),
                    Cell(x1=60, y1=10, x2=80, y2=22),
                ]
            ),
            Row(
                cells=[
                    Cell(x1=10, y1=22, x2=30, y2=37),
                    Cell(x1=30, y1=22, x2=60, y2=37),
                    Cell(x1=60, y1=22, x2=80, y2=37),
                ]
            ),
            Row(
                cells=[
                    Cell(x1=10, y1=37, x2=30, y2=52),
                    Cell(x1=30, y1=37, x2=60, y2=52),
                    Cell(x1=60, y1=37, x2=80, y2=52),
                ]
            ),
            Row(
                cells=[
                    Cell(x1=10, y1=52, x2=30, y2=65),
                    Cell(x1=30, y1=52, x2=60, y2=65),
                    Cell(x1=60, y1=52, x2=80, y2=65),
                ]
            ),
        ]
    )


def test_section_group_to_table_empty_group() -> None:
    with pytest.raises(ValueError, match="Empty group"):
        section_group_to_table(section_group=[])
