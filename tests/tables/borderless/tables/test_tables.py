from img2table.tables.borderless.tables import identify_tables
from img2table.tables.borderless.types import ColumnSection, MergedRow
from img2table.tables.types import Cell, Row, Table


def test_identify_tables(
    monkeypatch,  # noqa: ANN001
) -> None:
    expected = Table(rows=[Row(cells=[Cell(x1=0, y1=0, x2=10, y2=10)])])

    monkeypatch.setattr(
        "img2table.tables.borderless.tables.bridge_small_discrepancies",
        lambda structured_sections: [[structured_sections[0]]],
    )
    monkeypatch.setattr(
        "img2table.tables.borderless.tables.section_group_to_table",
        lambda section_gp: expected,  # noqa: ARG005
    )

    result = identify_tables(
        column_sections=[ColumnSection(rows=[MergedRow(items=[])])],
        char_length=5,
        height=100,
        width=100,
    )

    assert result == [expected]
