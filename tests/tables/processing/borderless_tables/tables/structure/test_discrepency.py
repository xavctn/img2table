from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables._model import ColumnSection, MergedRow
from img2table.tables.processing.borderless_tables.tables.structure.discrepency import (
    bridge_small_discrepencies,
)


def _column_section(y1: int, y2: int, columns: int = 3) -> ColumnSection:
    x_positions = [10, 40, 70, 100][:columns]
    rows = [
        MergedRow(
            items=[
                Cell(x1=x_pos, y1=y1, x2=x_pos + 10, y2=y1 + 10)
                for x_pos in x_positions
            ]
        ),
        MergedRow(
            items=[
                Cell(x1=x_pos, y1=y2 - 10, x2=x_pos + 10, y2=y2)
                for x_pos in x_positions
            ]
        ),
        MergedRow(
            items=[
                Cell(x1=x_pos, y1=y2 + 5, x2=x_pos + 10, y2=y2 + 15)
                for x_pos in x_positions
            ]
        ),
    ]
    return ColumnSection(
        items=[item for row in rows for item in row.items],
        rows=rows,
        whitespaces=rows[0].compute_whitespaces(min_width=10, x_min=0, x_max=120),
    )


def test_bridge_small_discrepencies_groups_matching_sections(
    monkeypatch,  # noqa: ANN001
) -> None:
    first = _column_section(y1=10, y2=30)
    second = _column_section(y1=35, y2=55)
    third = _column_section(y1=120, y2=140, columns=2)

    monkeypatch.setattr(
        "img2table.tables.processing.borderless_tables.tables.filter.model.StructuredSection.is_structured",
        lambda self: self.y_min < 100,
    )

    result = bridge_small_discrepencies(
        column_sections=[third, second, first],
        width=120,
        height=160,
        char_length=5,
    )

    assert [len(group) for group in result] == [2]
    assert [(sec.y_min, sec.y_max) for sec in result[0]] == [(10, 45), (35, 70)]

