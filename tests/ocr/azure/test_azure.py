import pytest

from img2table.document import Image
from img2table.ocr import AzureOCR
from tests.ocr_data_utils import read_ocr_data


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

    expected = read_ocr_data("test_data/ocr.csv")

    assert result == expected
