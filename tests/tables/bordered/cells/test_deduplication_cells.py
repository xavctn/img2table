from img2table.tables.bordered.cells.deduplication import deduplicate_cells
from tests.tables.bordered.cells import read_cells


def test_deduplicate_cells() -> None:
    cells = read_cells("test_data/expected_ident_cells.csv")

    result = deduplicate_cells(cells=cells)

    expected = read_cells("test_data/expected.csv")

    assert sorted(result, key=lambda c: (c.x1, c.y1, c.x2, c.y2)) == sorted(
        expected, key=lambda c: (c.x1, c.y1, c.x2, c.y2)
    )
