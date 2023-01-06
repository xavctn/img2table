# coding: utf-8
import pandas as pd

from img2table.tables.processing.cells.deduplication import deduplicate_cells_vertically, deduplicate_nested_cells, \
    deduplicate_cells


def test_deduplicate_cells_vertically():
    df_cells = pd.read_csv("test_data/expected_ident_cells.csv", sep=";", encoding="utf-8")

    result = deduplicate_cells_vertically(df_cells=df_cells)
    expected = pd.read_csv("test_data/expected_vertical_dedup.csv", sep=";", encoding="utf-8")
    expected.index = result.index

    assert result.equals(expected)


def test_deduplicate_nested_cells():
    df_cells = pd.read_csv("test_data/expected_vertical_dedup.csv", sep=";", encoding="utf-8")

    result = deduplicate_nested_cells(df_cells=df_cells)
    expected = pd.read_csv("test_data/expected.csv", sep=";", encoding="utf-8")
    expected.index = result.index

    assert result.equals(expected)


def test_deduplicate_cells():
    df_cells = pd.read_csv("test_data/expected_ident_cells.csv", sep=";", encoding="utf-8")

    result = deduplicate_cells(df_cells=df_cells)
    expected = pd.read_csv("test_data/expected.csv", sep=";", encoding="utf-8")
    expected.index = result.index

    assert result.equals(expected)
