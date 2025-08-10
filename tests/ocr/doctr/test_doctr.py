# coding: utf-8

import pickle
import sys

import polars as pl
import pytest

from img2table.document.image import Image
from img2table.ocr import DocTR
from img2table.ocr.data import OCRDataframe


def format_content(content):
    output = {
        id_page: {id_line: [{"value": word.value,
                             "confidence": round(word.confidence, 2),
                             "geometry": word.geometry,
                             }
                            for word in line.words]
                  for block in page.blocks for id_line, line in enumerate(block.lines)
                  }
        for id_page, page in enumerate(content.pages)
    }

    return output


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Not supported anymore")
def test_doctr_content():
    instance = DocTR()
    doc = Image(src="test_data/test.png")

    result = instance.content(document=doc)

    with open("test_data/ocr.pkl", "rb") as f:
        expected = pickle.load(f)

    assert format_content(result) == format_content(expected)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Not supported anymore")
def test_doctr_ocr_df():
    instance = DocTR()

    with open("test_data/ocr.pkl", "rb") as f:
        content = pickle.load(f)

    result = instance.to_ocr_dataframe(content=content)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Not supported anymore")
def test_doctr_document():
    instance = DocTR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result.df.drop("confidence").equals(expected.df.drop("confidence"))
