import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from img2table.document.image import Image
from img2table.ocr import EasyOCR
from img2table.ocr.data import OCRDataframe


def convert_np_types(obj: Any) -> Any:
    if isinstance(obj, list):
        return [convert_np_types(element) for element in obj]
    if isinstance(obj, dict):
        return {convert_np_types(k): convert_np_types(v) for k, v in obj.values()}
    if isinstance(obj, tuple):
        return [convert_np_types(element) for element in obj]
    if isinstance(obj, np.int32):
        return int(obj)
    if isinstance(obj, (np.float64, float)):
        return None
    return obj


def test_validators() -> None:
    with pytest.raises(TypeError):
        EasyOCR(lang=12)  # ty:ignore[invalid-argument-type]


def test_easyocr_content() -> None:
    instance = EasyOCR()
    doc = Image(src="test_data/test.png")

    result = instance.content(document=doc)

    with Path("test_data/ocr.json").open() as f:
        expected = json.load(f)

    assert convert_np_types(result) == convert_np_types(expected)


def test_easyocr_ocr_df() -> None:
    instance = EasyOCR()

    with Path("test_data/ocr.json").open() as f:
        content = json.load(f)

    result = instance.to_ocr_dataframe(content=content)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="Error building with 3.12")
def test_easyocr_document() -> None:
    instance = EasyOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result is not None
    assert result.df.drop("confidence").equals(expected.df.drop("confidence"))
