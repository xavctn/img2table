import pickle
import sys
from pathlib import Path

import polars as pl
import pytest

from img2table.document import Image
from img2table.ocr import SuryaOCR
from img2table.ocr.data import OCRDataframe
from tests import MOCK_DIR


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Library not available")
def test_content(mock_surya) -> None:  # noqa: ANN001, ARG001
    img = Image(src="test_data/test.png")
    ocr = SuryaOCR(langs=["en"])

    result = ocr.content(document=img)

    with (Path(MOCK_DIR) / "surya.pkl").open("rb") as f:
        expected = pickle.load(f)

    assert result == expected


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Library not available")
def test_to_ocr_df() -> None:
    ocr = SuryaOCR(langs=["en"])
    with (Path(MOCK_DIR) / "surya.pkl").open("rb") as f:
        content = pickle.load(f)

    result = ocr.to_ocr_dataframe(content=content)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Library not available")
def test_surya_ocr(mock_surya) -> None:  # noqa: ANN001, ARG001
    # Test init error
    with pytest.raises(TypeError):
        SuryaOCR(langs=1)  # ty:ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        SuryaOCR(langs=[1, 2])  # ty:ignore[invalid-argument-type]

    img = Image(src="test_data/test.png")
    ocr = SuryaOCR(langs=["en"])

    result = ocr.of(document=img)

    expected = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    assert result == expected
