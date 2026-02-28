import json
import os
import pickle
from pathlib import Path

import polars as pl
import pytest

from img2table.document import Image
from img2table.ocr.data import OCRDataframe
from img2table.ocr.google_vision import VisionAPIContent, VisionEndpointContent, VisionOCR
from tests import MOCK_DIR


def test_vision_endpoint_content(mock_vision) -> None:  # noqa: ANN001, ARG001
    image = Image(src="test_data/test.png")
    content = VisionEndpointContent(api_key="api_key", timeout=10)

    with Path("test_data/expected_content.json").open() as f:
        expected = json.load(f)

    # Test for map_response method
    with (Path(MOCK_DIR) / "vision.json").open() as f:
        response = json.load(f)

    result_map_response = content.map_response(response=response, page=0, height=417, width=1365)
    assert result_map_response == expected[0]

    # Test for get_content method
    result_get_content = content.get_content(document=image)
    assert result_get_content == expected


def test_vision_api_content(mock_vision) -> None:  # noqa: ANN001, ARG001
    image = Image(src="test_data/test.png")
    content = VisionAPIContent(timeout=10)

    with Path("test_data/expected_content.json").open() as f:
        expected = json.load(f)

    # Test for map_response method
    with (Path(MOCK_DIR) / "vision.pkl").open("rb") as f:
        response = pickle.load(f)

    result_map_response = content.map_response(response=response, shapes=[(417, 1365)])
    assert result_map_response == expected

    # Test for get_content method
    result_get_content = content.get_content(document=image)
    assert result_get_content == expected


def test_vision_ocr(mock_vision) -> None:  # noqa: ANN001, ARG001
    image = Image(src="test_data/test.png")

    with Path("test_data/expected_content.json").open() as f:
        content = json.load(f)

    expected_ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";"))

    # Test init error
    with pytest.raises(TypeError):
        VisionOCR(api_key=8)  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValueError):
        VisionOCR()

    # Test with api_key
    ocr_key = VisionOCR(timeout=10, api_key="api_key")

    result_to_ocr_df = ocr_key.to_ocr_dataframe(content=content)
    assert result_to_ocr_df == expected_ocr_df

    result_ocr_df = ocr_key.of(document=image)
    assert result_ocr_df == expected_ocr_df

    # Test with credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds"
    ocr_creds = VisionOCR(timeout=10)

    result_to_ocr_df = ocr_creds.to_ocr_dataframe(content=content)
    assert result_to_ocr_df == expected_ocr_df

    result_ocr_df = ocr_creds.of(document=image)
    assert result_ocr_df == expected_ocr_df
