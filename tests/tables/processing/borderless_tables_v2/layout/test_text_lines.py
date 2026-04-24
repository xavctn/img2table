import json
from pathlib import Path

import cv2

from img2table.tables import threshold_dark_areas
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables_v2.layout.text_lines import (
    identify_image_contours,
)


def test_identify_image_contours() -> None:
    img = cv2.cvtColor(cv2.imread("test_data/test.bmp"), cv2.COLOR_BGR2RGB)
    thresh = threshold_dark_areas(img=img, char_length=6)

    with Path("test_data/lines.json").open() as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get("h_lines") + data.get("v_lines")]

    result = identify_image_contours(thresh=thresh, lines=lines, char_length=6.0)

    with Path("test_data/contours.json").open() as f:
        expected = [Cell(**row) for row in json.load(f)]
    assert result == expected
