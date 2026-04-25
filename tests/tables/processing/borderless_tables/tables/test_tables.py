from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables._model import ColumnSection, MergedRow
from img2table.tables.processing.borderless_tables.tables import identify_tables


def test_identify_tables(
    monkeypatch,  # noqa: ANN001
) -> None:
    expected = Table(rows=[Row(cells=[Cell(x1=0, y1=0, x2=10, y2=10)])])

    monkeypatch.setattr(
        "img2table.tables.processing.borderless_tables.tables.bridge_small_discrepencies",
        lambda column_sections, char_length, height, width: [[column_sections[0]]],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "img2table.tables.processing.borderless_tables.tables.section_group_to_table",
        lambda section_gp: expected,  # noqa: ARG005
    )

    result = identify_tables(
        column_sections=[ColumnSection(rows=[MergedRow(items=[])])],
        char_length=5,
        height=100,
        width=100,
    )

    assert result == [expected]
