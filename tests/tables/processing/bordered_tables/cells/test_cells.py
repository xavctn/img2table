import json
from pathlib import Path

from img2table.tables.objects.line import Line
from img2table.tables.processing.bordered_tables.cells import get_cells
from tests.tables.processing.bordered_tables.cells import read_cells


def test_get_cells() -> None:
    with Path("test_data/lines.json").open() as f:
        data = json.load(f)
    h_lines = [Line(**el) for el in data.get("h_lines")]
    v_lines = [Line(**el) for el in data.get("v_lines")]

    result = get_cells(horizontal_lines=h_lines, vertical_lines=v_lines)

    expected = read_cells("test_data/expected.csv")

    assert sorted(result, key=lambda c: (c.x1, c.y1, c.x2, c.y2)) == sorted(
        expected, key=lambda c: (c.x1, c.y1, c.x2, c.y2)
    )
