# coding: utf-8
import json
import os

import polars as pl

from img2table.document import Image
from img2table.ocr import TextractOCR
from img2table.ocr.data import OCRDataframe
from tests import MOCK_DIR


def test_map_response(mock_textract):
    img = Image("test_data/test.png")

    with open(os.path.join(MOCK_DIR, "textract.json"), "r") as f:
        resp = json.load(f)

    result = TextractOCR().map_response(response=resp,
                                        image=list(img.images)[0],
                                        page=0)

    with open("test_data/content.json", "r") as f:
        expected = json.load(f)

    assert result == expected


def test_content(mock_textract):
    img = Image("test_data/test.png")
    ocr = TextractOCR()

    result = ocr.content(document=img)

    with open("test_data/content.json", "r") as f:
        expected = json.load(f)

    assert list(result) == [expected]


def test_to_ocr_df(mock_textract):
    ocr = TextractOCR()
    with open("test_data/content.json", "r") as f:
        content = json.load(f)

    result = ocr.to_ocr_dataframe(content=[content])

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    assert result == expected


def test_textract_ocr(mock_textract):
    img = Image("test_data/test.png")
    ocr = TextractOCR(aws_access_key_id="aws_access_key_id",
                      aws_secret_access_key="aws_secret_access_key",
                      aws_session_token="aws_session_token",
                      region="eu-west-1")

    result = ocr.of(document=img)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    assert result == expected
