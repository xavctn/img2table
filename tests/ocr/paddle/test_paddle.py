# coding: utf-8

import json
import sys

import cv2
import polars as pl
import pytest

from img2table.document.image import Image
from img2table.ocr.data import OCRDataframe


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="No support for PaddleOCR")
def test_validators():
    from img2table.ocr import PaddleOCR

    with pytest.raises(TypeError) as e_info:
        ocr = PaddleOCR(lang=12)


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="No support for PaddleOCR")
def test_paddle_hocr():
    from img2table.ocr import PaddleOCR

    instance = PaddleOCR()
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    result = instance.hocr(image=img)

    with open("test_data/hocr.json", "r") as f:
        expected = [[element[0], tuple(element[1])] for element in json.load(f)]

    assert result == expected


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="No support for PaddleOCR")
def test_paddle_content():
    from img2table.ocr import PaddleOCR

    instance = PaddleOCR()
    doc = Image(src="test_data/test.png")

    result = instance.content(document=doc)

    with open("test_data/hocr.json", "r") as f:
        expected = [[[element[0], tuple(element[1])] for element in json.load(f)]]

    assert result == expected


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="No support for PaddleOCR")
def test_paddle_ocr_df():
    from img2table.ocr import PaddleOCR

    instance = PaddleOCR()

    with open("test_data/hocr.json", "r") as f:
        content = [[[element[0], tuple(element[1])] for element in json.load(f)]]

    result = instance.to_ocr_dataframe(content=content)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";").lazy())

    assert result == expected


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="No support for PaddleOCR")
def test_paddle_document():
    from img2table.ocr import PaddleOCR

    instance = PaddleOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";").lazy())

    assert result == expected
