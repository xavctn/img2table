# coding: utf-8
import polars as pl

from img2table.tables.processing.cells.deduplication import deduplicate_cells_vertically, deduplicate_nested_cells, \
    deduplicate_cells


def test_deduplicate_cells_vertically():
    df_cells = pl.read_csv("test_data/expected_ident_cells.csv", sep=";", encoding="utf-8").lazy()

    result = deduplicate_cells_vertically(df_cells=df_cells).collect()
    expected = pl.read_csv("test_data/expected_vertical_dedup.csv", sep=";", encoding="utf-8")

    assert result.frame_equal(expected)


def test_deduplicate_nested_cells():
    df_cells = pl.read_csv("test_data/expected_vertical_dedup.csv", sep=";", encoding="utf-8").lazy()

    result = deduplicate_nested_cells(df_cells=df_cells).collect()
    expected = pl.read_csv("test_data/expected.csv", sep=";", encoding="utf-8")

    assert result.frame_equal(expected)


def test_deduplicate_cells():
    df_cells = pl.read_csv("test_data/expected_ident_cells.csv", sep=";", encoding="utf-8").lazy()

    result = deduplicate_cells(df_cells=df_cells).collect()
    expected = pl.read_csv("test_data/expected.csv", sep=";", encoding="utf-8")

    assert result.frame_equal(expected)
