import cv2
import numpy as np

from img2table.tables.borderless.tables.filter.graphics import is_graphic_region
from img2table.tables.types import Cell, Line, Row, Table


def test_graphic_region_rejects_plot_and_keeps_text_table() -> None:
    table = Table(
        rows=[
            Row(cells=[Cell(x1=10, y1=y, x2=190, y2=y + 20)])
            for y in (20, 40, 60)
        ]
    )
    plot = np.zeros((120, 200), dtype=np.uint8)
    cv2.line(plot, (40, 0), (40, 110), (255,), 1)
    for y in (30, 50, 70):
        cv2.line(plot, (10, y), (190, y), (255,), 1)

    text = np.zeros_like(plot)
    for y in (35, 55, 75):
        for x in (20, 80, 140):
            cv2.putText(text, "AB", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,), 1)

    assert is_graphic_region(plot, table, char_length=8, lines=[Line(40, 0, 40, 110)])
    assert not is_graphic_region(text, table, char_length=8, lines=[])
