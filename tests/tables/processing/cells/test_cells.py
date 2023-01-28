# coding: utf-8
import json

import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.cells import get_cells


def test_get_cells():
    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    h_lines = [Line(**el) for el in data.get('h_lines')]
    v_lines = [Line(**el) for el in data.get('v_lines')]

    result = get_cells(horizontal_lines=h_lines,
                       vertical_lines=v_lines)

    df_expected = pl.read_csv("test_data/expected.csv", sep=";", encoding="utf-8")
    expected = [Cell(x1=row["x1"], x2=row["x2"], y1=row["y1"], y2=row["y2"])
                for row in df_expected.to_dicts()]

    assert result == expected
