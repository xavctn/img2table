from img2table.tables.borderless.tables.filter.model import StructuredSection
from img2table.tables.borderless.tables.structure.discrepancy import (
    bridge_small_discrepancies,
)
from img2table.tables.borderless.types import MergedRow
from img2table.tables.types import Cell


def _structured_section(y1: int, y2: int, idx: int, columns: int = 3) -> StructuredSection:
    x_positions = [10, 40, 70, 100][:columns]
    rows = [
        MergedRow(
            items=[Cell(x1=x_pos, y1=y1, x2=x_pos + 10, y2=y1 + 10) for x_pos in x_positions]
        ),
        MergedRow(
            items=[Cell(x1=x_pos, y1=y2 - 10, x2=x_pos + 10, y2=y2) for x_pos in x_positions]
        ),
        MergedRow(
            items=[Cell(x1=x_pos, y1=y2 + 5, x2=x_pos + 10, y2=y2 + 15) for x_pos in x_positions]
        ),
    ]
    return StructuredSection(
        idx=idx,
        height=160,
        width=120,
        char_length=5,
        items=[item for row in rows for item in row.items],
        merged_rows=rows,
        whitespaces=rows[0].compute_whitespaces(min_width=10, x_min=0, x_max=120),
    )


def test_bridge_small_discrepancies_groups_matching_sections(
    monkeypatch,  # noqa: ANN001
) -> None:
    first = _structured_section(y1=10, y2=30, idx=0)
    second = _structured_section(y1=35, y2=55, idx=1)
    third = _structured_section(y1=120, y2=140, idx=2, columns=2)

    monkeypatch.setattr(
        "img2table.tables.borderless.tables.filter.model.StructuredSection.is_structured",
        lambda self: self.y_min < 100,
    )

    result = bridge_small_discrepancies(
        structured_sections=[third, second, first],
    )

    assert [len(group) for group in result] == [2]
    assert [(sec.y_min, sec.y_max) for sec in result[0]] == [(10, 45), (35, 70)]
