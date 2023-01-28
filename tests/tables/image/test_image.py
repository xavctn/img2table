# coding: utf-8
import json

import cv2
import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.image import TableImage
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table


def test_table_image():
    image = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(pl.read_csv("test_data/ocr.csv", sep=";", encoding="utf-8").lazy())

    tb_image = TableImage(img=image,
                          dpi=200,
                          ocr_df=ocr_df,
                          min_confidence=50)

    result = tb_image.extract_tables(implicit_rows=True)

    with open("test_data/expected_tables.json", "r") as f:
        expected = [Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in tb]).extracted_table
                    for tb in json.load(f)]

    assert result == expected
