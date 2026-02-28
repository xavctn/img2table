import json
from pathlib import Path
from typing import Any

import polars as pl

from img2table.document import Image
from img2table.ocr import TextractOCR
from img2table.ocr.data import OCRDataframe
from tests import MOCK_DIR


def test_map_response(mock_textract) -> None:  # noqa: ANN001, ARG001
    img = Image(src="test_data/test.png")

    with (Path(MOCK_DIR) / "textract.json").open() as f:
        resp = json.load(f)

    result = TextractOCR().map_response(response=resp, image=next(iter(img.images)), page=0)

    with (Path("test_data") / "content.json").open() as f:
        expected = json.load(f)

    assert result == expected


def test_content(mock_textract) -> None:  # noqa: ANN001, ARG001
    img = Image(src="test_data/test.png")
    ocr = TextractOCR()

    result = ocr.content(document=img)

    with (Path("test_data") / "content.json").open() as f:
        expected = json.load(f)

    assert list(result) == [expected]


def test_to_ocr_df(mock_textract) -> None:  # noqa: ANN001, ARG001
    ocr = TextractOCR()
    with (Path("test_data") / "content.json").open() as f:
        content: dict[str, Any] = json.load(f)

    result = ocr.to_ocr_dataframe(content=[content])  # ty:ignore[invalid-argument-type]

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected


def test_textract_ocr(mock_textract) -> None:  # noqa: ANN001, ARG001
    img = Image(src="test_data/test.png")
    ocr = TextractOCR(
        aws_access_key_id="aws_access_key_id",
        aws_secret_access_key="aws_secret_access_key",
        aws_session_token="aws_session_token",
        region="eu-west-1",
    )

    result = ocr.of(document=img)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected
