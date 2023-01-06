# coding: utf-8
import json

import pandas as pd

from img2table.tables.objects.line import Line
from img2table.tables.processing.cells.identification import get_cells_dataframe


def test_get_cells_dataframe():
    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    h_lines = [Line(**el) for el in data.get('h_lines')]
    v_lines = [Line(**el) for el in data.get('v_lines')]

    result = get_cells_dataframe(horizontal_lines=h_lines,
                                 vertical_lines=v_lines)
    expected = pd.read_csv("test_data/expected_ident_cells.csv", sep=";", encoding="utf-8")

    assert result.equals(expected)
