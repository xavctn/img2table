# coding: utf-8
import json

import polars as pl

from img2table.document.pdf import PDF
from img2table.ocr.data import OCRDataframe
from img2table.ocr.pdf import PdfOCR


def test_pdf_content(mock_tesseract):
    instance = PdfOCR()
    doc = PDF(src="test_data/test.pdf", dpi=300)

    result = instance.content(document=doc)

    with open("test_data/content.json", "r") as f:
        expected = json.load(f)

    assert result == expected


def test_pdf_ocr_df():
    instance = PdfOCR()

    with open("test_data/content.json", "r") as f:
        content = json.load(f)

    result = instance.to_ocr_dataframe(content=content)

    df_expected = pl.read_csv("test_data/ocr_df.csv", sep=";").lazy()
    expected = OCRDataframe(df=df_expected)

    assert result == expected


def test_pdf_document():
    instance = PdfOCR()
    doc = PDF(src="test_data/test.pdf", dpi=300)

    result = instance.of(document=doc)

    df_expected = pl.read_csv("test_data/ocr_df.csv", sep=";").lazy()
    expected = OCRDataframe(df=df_expected)

    assert result == expected
