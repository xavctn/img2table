import sys

import pytest

from img2table.document.image import Image
from tests.ocr_data_utils import read_ocr_data

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Paddle unsupported on Python 3.14+"
)


def test_validators() -> None:
    from img2table.ocr import PaddleOCR

    with pytest.raises(TypeError):
        PaddleOCR(lang=12)  # ty:ignore[invalid-argument-type]


def test_paddle_document() -> None:
    from img2table.ocr import PaddleOCR

    instance = PaddleOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = read_ocr_data("test_data/ocr.csv")

    assert result == expected
