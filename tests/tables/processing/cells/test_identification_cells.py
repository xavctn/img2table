# coding: utf-8
import pandas as pd

from img2table.tables.objects.line import Line
from img2table.tables.processing.cells import get_cells_dataframe


def test_get_cells_dataframe():
    h_lines = [Line(x1=157, y1=92, x2=835, y2=92),
               Line(x1=157, y1=168, x2=835, y2=168),
               Line(x1=157, y1=212, x2=835, y2=212),
               Line(x1=157, y1=256, x2=835, y2=256),
               Line(x1=157, y1=299, x2=835, y2=299),
               Line(x1=157, y1=342, x2=835, y2=342),
               Line(x1=157, y1=386, x2=835, y2=386),
               Line(x1=157, y1=430, x2=835, y2=430),
               Line(x1=156, y1=473, x2=836, y2=473)]

    v_lines = [Line(x1=156, y1=92, x2=156, y2=475),
               Line(x1=434, y1=92, x2=434, y2=474),
               Line(x1=587, y1=92, x2=587, y2=474),
               Line(x1=834, y1=92, x2=834, y2=475)]

    result = get_cells_dataframe(horizontal_lines=h_lines,
                                 vertical_lines=v_lines)
    expected = pd.read_csv("test_data/expected_ident_cells.csv", sep=";", encoding="utf-8")

    assert result.equals(expected)
