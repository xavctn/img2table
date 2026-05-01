import json
from pathlib import Path

import cv2

from img2table.tables.bordered.lines import detect_lines
from img2table.tables.types import Cell, Line


def test_detect_lines() -> None:
    img = cv2.imread("test_data/test.png")
    assert img is not None
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    with Path("test_data/contours.json").open() as f:
        contours = [Cell(**el) for el in json.load(f)]

    h_lines, v_lines = detect_lines(
        img=img, contours=contours, char_length=8.85, min_line_length=10
    )

    with Path("test_data/expected.json").open() as f:
        data = json.load(f)
    h_lines_expected = [Line(**el) for el in data.get("h_lines")]
    v_lines_expected = [Line(**el) for el in data.get("v_lines")]

    h_lines = sorted(h_lines, key=lambda ln: (ln.x1, ln.y1, ln.x2, ln.y2))
    v_lines = sorted(v_lines, key=lambda ln: (ln.x1, ln.y1, ln.x2, ln.y2))
    h_lines_expected = sorted(h_lines_expected, key=lambda ln: (ln.x1, ln.y1, ln.x2, ln.y2))
    v_lines_expected = sorted(v_lines_expected, key=lambda ln: (ln.x1, ln.y1, ln.x2, ln.y2))

    assert (h_lines, v_lines) == (h_lines_expected, v_lines_expected)
