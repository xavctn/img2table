import pytest

from img2table.document import Image
from img2table.ocr import SuryaOCR
from tests.ocr_data_utils import read_ocr_data


def test_surya_ocr(mock_surya) -> None:  # noqa: ANN001, ARG001
    # Test init error
    with pytest.raises(TypeError):
        SuryaOCR(langs=1)  # ty:ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        SuryaOCR(langs=[1, 2])  # ty:ignore[invalid-argument-type]

    img = Image(src="test_data/test.png")
    ocr = SuryaOCR(langs=["en"])

    result = ocr.of(document=img)

    expected = read_ocr_data("test_data/ocr.csv")

    assert result == expected
