import pickle
from pathlib import Path

import polars as pl
import pytest

from img2table.document import Image
from img2table.ocr import AzureOCR
from img2table.ocr.data import OCRDataframe
from tests import MOCK_DIR


def test_content(mock_azure) -> None:  # noqa: ANN001, ARG001
    img = Image(src="test_data/test.png")
    ocr = AzureOCR(endpoint="aa", subscription_key="bb")

    result = ocr.content(document=img)

    with (Path(MOCK_DIR) / "azure.pkl").open("rb") as f:
        expected = pickle.load(f)

    assert list(result) == [expected]


def test_to_ocr_df(mock_azure) -> None:  # noqa: ANN001, ARG001
    ocr = AzureOCR(endpoint="aa", subscription_key="bb")
    with (Path(MOCK_DIR) / "azure.pkl").open("rb") as f:
        content = pickle.load(f)

    result = ocr.to_ocr_dataframe(content=[content])

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected


def test_azure_ocr(mock_azure) -> None:  # noqa: ANN001, ARG001
    # Test init error
    with pytest.raises(TypeError):
        AzureOCR(subscription_key=8, endpoint="a")  # ty:ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        AzureOCR(subscription_key="a", endpoint=0)  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValueError):
        AzureOCR(subscription_key="a")

    with pytest.raises(ValueError):
        AzureOCR(endpoint="a")

    img = Image(src="test_data/test.png")
    ocr = AzureOCR(endpoint="aa", subscription_key="bb")

    result = ocr.of(document=img)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected
