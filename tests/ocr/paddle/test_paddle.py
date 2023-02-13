# coding: utf-8

import json

import cv2
import polars as pl
import pytest

from img2table.document.image import Image
from img2table.ocr import PaddleOCR
from img2table.ocr.data import OCRDataframe


def test_validators():
    with pytest.raises(TypeError) as e_info:
        ocr = PaddleOCR(lang=12)


def test_paddle_hocr():
    instance = PaddleOCR()
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    result = instance.hocr(image=img)

    with open("test_data/hocr.json", "r") as f:
        expected = [[element[0], tuple(element[1])] for element in json.load(f)]

    assert result == expected


def test_paddle_content():
    instance = PaddleOCR()
    doc = Image(src="test_data/test.png")

    result = instance.content(document=doc)

    with open("test_data/hocr.json", "r") as f:
        expected = [[[element[0], tuple(element[1])] for element in json.load(f)]]

    assert result == expected


def test_paddle_ocr_df():
    instance = PaddleOCR()

    with open("test_data/hocr.json", "r") as f:
        content = [[[element[0], tuple(element[1])] for element in json.load(f)]]

    result = instance.to_ocr_dataframe(content=content)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    assert result == expected


def test_paddle_document():
    instance = PaddleOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    assert result == expected
