# coding: utf-8
import json

import cv2
import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.text.titles import get_title_tables


def test_get_title_tables():
    img = cv2.imread("test_data/test.jpg", cv2.IMREAD_GRAYSCALE)
    with open("test_data/table.json", "r") as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr.csv", sep=";").lazy())

    result = get_title_tables(img=img, tables=[table], ocr_df=ocr_df)

    assert result[0].title == "10 most populous countries"
    assert get_title_tables(img=img, tables=[], ocr_df=ocr_df) == []
