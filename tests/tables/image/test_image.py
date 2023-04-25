# coding: utf-8

import cv2
import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.image import TableImage
from img2table.tables.objects.extraction import BBox


def test_table_image():
    image = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(pl.read_csv("test_data/ocr.csv", separator=";", encoding="utf-8").lazy())

    tb_image = TableImage(img=image,
                          dpi=200,
                          ocr_df=ocr_df,
                          min_confidence=50)

    result = tb_image.extract_tables(implicit_rows=True)

    assert result[0].bbox == BBox(x1=35, y1=20, x2=770, y2=327)
    assert (len(result[0].content), len(result[0].content[0])) == (6, 3)

    assert result[1].bbox == BBox(x1=962, y1=20, x2=1154, y2=123)
    assert (len(result[1].content), len(result[1].content[0])) == (2, 2)
