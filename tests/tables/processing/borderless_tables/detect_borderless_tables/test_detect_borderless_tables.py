# coding: utf-8

import json

import cv2
import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables import detect_borderless_tables, deduplicate_tables


def test_deduplicate_tables():
    identified_tables = [
        Table(rows=Row(cells=[Cell(x1=0, x2=100, y1=0, y2=100)])),
        Table(rows=Row(cells=[Cell(x1=200, x2=300, y1=200, y2=300)])),
        Table(rows=Row(cells=[Cell(x1=200, x2=300, y1=200, y2=250)])),
    ]

    existing_tables = [Table(rows=Row(cells=[Cell(x1=0, x2=100, y1=0, y2=100)]))]

    result = deduplicate_tables(identified_tables=identified_tables, existing_tables=existing_tables)

    assert result == [Table(rows=Row(cells=[Cell(x1=200, x2=300, y1=200, y2=300)]))]


def test_detect_borderless_tables():
    img = cv2.imread("test_data/test.jpg", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    result = detect_borderless_tables(img=img, ocr_df=ocr_df, existing_tables=[])

    with open("test_data/expected.json", "r") as f:
        expected = [Table(rows=[Row(cells=[Cell(**c) for c in row]) for row in tb]) for tb in json.load(f)]

    assert result == expected
