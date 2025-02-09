# coding: utf-8

import json
import sys
from typing import Any

import numpy as np
import polars as pl
import pytest

from img2table.document.image import Image
from img2table.ocr import EasyOCR
from img2table.ocr.data import OCRDataframe


def convert_np_types(obj: Any):
    if isinstance(obj, list):
        return [convert_np_types(element) for element in obj]
    elif isinstance(obj, dict):
        return {convert_np_types(k): convert_np_types(v) for k, v in obj.values()}
    elif isinstance(obj, tuple):
        return list(convert_np_types(element) for element in obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, (np.float64, float)):
        return None
    else:
        return obj


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="Error building with 3.12")
def test_validators():
    with pytest.raises(TypeError) as e_info:
        ocr = EasyOCR(lang=12)


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="Error building with 3.12")
def test_easyocr_content():
    instance = EasyOCR()
    doc = Image(src="test_data/test.png")

    result = instance.content(document=doc)

    with open("test_data/ocr.json", "r") as f:
        expected = json.load(f)

    assert convert_np_types(result) == convert_np_types(expected)


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="Error building with 3.12")
def test_easyocr_ocr_df():
    instance = EasyOCR()

    with open("test_data/ocr.json", "r") as f:
        content = json.load(f)

    result = instance.to_ocr_dataframe(content=content)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="Error building with 3.12")
def test_easyocr_document():
    instance = EasyOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result.df.drop("confidence").equals(expected.df.drop("confidence"))
