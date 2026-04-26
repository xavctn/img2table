import os

import pytest

from img2table.document import Image
from img2table.ocr.google_vision import VisionAPIContent, VisionEndpointContent, VisionOCR
from tests.ocr_data_utils import read_ocr_data


def test_vision_endpoint_content(mock_vision) -> None:  # noqa: ANN001, ARG001
    image = Image(src="test_data/test.png")
    instance = VisionEndpointContent(api_key="api_key", timeout=10)

    # Test for get_content method
    result = instance.of(document=image)

    expected = read_ocr_data("test_data/ocr.csv")
    assert result == expected


def test_vision_api_content(mock_vision) -> None:  # noqa: ANN001, ARG001
    image = Image(src="test_data/test.png")
    instance = VisionAPIContent(timeout=10)

    # Test for get_content method
    result = instance.of(document=image)

    expected = read_ocr_data("test_data/ocr.csv")
    assert result == expected


def test_vision_ocr(mock_vision) -> None:  # noqa: ANN001, ARG001
    image = Image(src="test_data/test.png")
    expected = read_ocr_data("test_data/ocr.csv")

    # Test init error
    with pytest.raises(TypeError):
        VisionOCR(api_key=8)  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValueError):
        VisionOCR()

    # Test with api_key
    ocr_key = VisionOCR(timeout=10, api_key="api_key")

    result = ocr_key.of(document=image)

    assert result == expected

    # Test with credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds"
    ocr_creds = VisionOCR(timeout=10)

    result = ocr_creds.of(document=image)

    assert result == expected
