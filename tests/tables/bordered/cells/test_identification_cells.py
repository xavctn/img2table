import json
from pathlib import Path

from img2table.tables.bordered.cells.identification import get_cells_from_lines
from img2table.tables.types import Line
from tests.tables.bordered.cells import read_cells


def test_get_cells_dataframe() -> None:
    with Path("test_data/lines.json").open() as f:
        data = json.load(f)
    h_lines = [Line(**el) for el in data.get("h_lines")]
    v_lines = [Line(**el) for el in data.get("v_lines")]

    result = get_cells_from_lines(horizontal_lines=h_lines, vertical_lines=v_lines)

    expected = read_cells("test_data/expected_ident_cells.csv")

    assert sorted(result, key=lambda c: (c.x1, c.y1, c.x2, c.y2)) == sorted(
        expected, key=lambda c: (c.x1, c.y1, c.x2, c.y2)
    )
