# coding: utf-8
import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.bordered_tables.cells.deduplication import deduplicate_cells


def test_deduplicate_cells():
    df_cells = pl.read_csv("test_data/expected_ident_cells.csv", separator=";", encoding="utf-8")
    cells = [Cell(x1=row["x1"], x2=row["x2"], y1=row["y1"], y2=row["y2"])
             for row in df_cells.to_dicts()]

    result = deduplicate_cells(cells=cells)

    df_expected = pl.read_csv("test_data/expected.csv", separator=";", encoding="utf-8")
    expected = [Cell(x1=row["x1"], x2=row["x2"], y1=row["y1"], y2=row["y2"])
                for row in df_expected.to_dicts()]

    assert sorted(result, key=lambda c: (c.x1, c.y1, c.x2, c.y2)) == sorted(expected, key=lambda c: (c.x1, c.y1, c.x2, c.y2))
