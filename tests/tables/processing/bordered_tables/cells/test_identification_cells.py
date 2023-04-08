# coding: utf-8
import json

import polars as pl

from img2table.tables.objects.line import Line
from img2table.tables.processing.bordered_tables.cells.identification import get_cells_dataframe, \
    get_potential_cells_from_h_lines


def test_get_potential_cells_from_h_lines():
    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    h_lines = [Line(**el) for el in data.get('h_lines')]

    df_h_lines = pl.from_dicts([l.dict for l in h_lines]).lazy()

    result = get_potential_cells_from_h_lines(df_h_lines=df_h_lines).collect()

    expected = pl.read_csv("test_data/expected_potential_cells.csv", separator=";", encoding="utf-8")

    assert result.frame_equal(expected)


def test_get_cells_dataframe():
    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    h_lines = [Line(**el) for el in data.get('h_lines')]
    v_lines = [Line(**el) for el in data.get('v_lines')]

    result = get_cells_dataframe(horizontal_lines=h_lines,
                                 vertical_lines=v_lines).collect()
    expected = pl.read_csv("test_data/expected_ident_cells.csv", separator=";", encoding="utf-8")

    assert result.frame_equal(expected.sort(['x1', 'y1', 'x2', 'y2']))
