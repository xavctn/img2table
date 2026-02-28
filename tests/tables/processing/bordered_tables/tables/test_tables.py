import json
from pathlib import Path

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.tables import get_tables


def test_get_tables() -> None:
    with Path("test_data/cells.json").open() as f:
        cells = [Cell(**el) for el in json.load(f)]
    with Path("test_data/contours.json").open() as f:
        contours = [Cell(**el) for el in json.load(f)]
    with Path("test_data/lines.json").open() as f:
        data = json.load(f)
        lines = [Line(**el) for el in data.get("h_lines") + data.get("v_lines")]

    result = get_tables(cells=cells, elements=contours, lines=lines, char_length=8.44)

    with Path("test_data/expected.json").open() as f:
        expected = [
            Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb]) for tb in json.load(f)
        ]

    assert result == expected
