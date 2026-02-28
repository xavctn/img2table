import json
from pathlib import Path

import polars as pl

from img2table.document.pdf import PDF
from img2table.ocr.data import OCRDataframe
from img2table.ocr.pdf import PdfOCR


def test_pdf_content(mock_tesseract) -> None:  # noqa: ANN001, ARG001
    instance = PdfOCR()
    doc = PDF(src="test_data/test.pdf", pages=[0, 1])

    result = instance.content(document=doc)

    with Path("test_data/content.json").open() as f:
        expected = json.load(f)

    assert result == expected


def test_pdf_ocr_df() -> None:
    instance = PdfOCR()

    with Path("test_data/content.json").open() as f:
        content = json.load(f)

    result = instance.to_ocr_dataframe(content=content)

    df_expected = pl.read_csv("test_data/ocr_df.csv", separator=";")
    expected = OCRDataframe(df=df_expected)

    assert result == expected


def test_pdf_document() -> None:
    instance = PdfOCR()
    doc = PDF(src="test_data/test.pdf", pages=[0, 1])

    result = instance.of(document=doc)

    df_expected = pl.read_csv("test_data/ocr_df.csv", separator=";")
    expected = OCRDataframe(df=df_expected)

    assert result == expected
