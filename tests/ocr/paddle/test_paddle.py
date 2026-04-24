import json
import sys
from pathlib import Path

import polars as pl
import pytest

from img2table.document.image import Image
from img2table.ocr.data import OCRDataframe
from tests.conftest import nested_approx

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Paddle unsupported on Python 3.14+"
)


def test_validators() -> None:
    from img2table.ocr import PaddleOCR

    with pytest.raises(TypeError):
        PaddleOCR(lang=12)  # ty:ignore[invalid-argument-type]


def test_paddle_content() -> None:
    from img2table.ocr import PaddleOCR

    instance = PaddleOCR()
    doc = Image(src="test_data/test.png")

    result = instance.content(document=doc)

    with Path("test_data/hocr.json").open() as f:
        expected = json.load(f)

    assert result == nested_approx(expected, abs=1e-3)


def test_paddle_ocr_df() -> None:
    from img2table.ocr import PaddleOCR

    instance = PaddleOCR()

    with Path("test_data/hocr.json").open() as f:
        content = json.load(f)

    result = instance.to_ocr_dataframe(content=content)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected


def test_paddle_document() -> None:
    from img2table.ocr import PaddleOCR

    instance = PaddleOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected
