# coding: utf-8
import json

import cv2
import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables import identify_borderless_tables


def test_identify_borderless_tables():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";").lazy())

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = identify_borderless_tables(img=img,
                                        ocr_df=ocr_df,
                                        lines=lines,
                                        existing_tables=[])

    assert len(result) == 2
    assert (result[0].nb_columns, result[0].nb_rows) == (3, 6)
    assert (result[1].nb_columns, result[1].nb_rows) == (2, 2)
