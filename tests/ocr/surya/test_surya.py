import pytest

from img2table.document import Image
from img2table.ocr import SuryaOCR
from img2table.ocr._types import OCRData


def test_surya_ocr(mock_surya) -> None:  # noqa: ANN001, ARG001
    # Test init error
    with pytest.raises(TypeError):
        SuryaOCR(langs=1)  # ty:ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        SuryaOCR(langs=[1, 2])  # ty:ignore[invalid-argument-type]

    img = Image(src="test_data/test.png")
    ocr = SuryaOCR(langs=["en"])

    result = ocr.of(document=img)

    expected = OCRData(
        records={
            0: [
                {
                    "id": "word_1_1_0", "parent": "word_1_1", "value": "Title",
                    "confidence": 100, "x1": 10, "y1": 20, "x2": 50, "y2": 40,
                },
                {
                    "id": "word_1_2_0", "parent": "word_1_2", "value": "Line 1",
                    "confidence": 95, "x1": 60, "y1": 20, "x2": 120, "y2": 40,
                },
            ]
        }
    )

    assert result == expected
