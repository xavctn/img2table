import pytest

from img2table.document.image import Image
from img2table.ocr import EasyOCR
from tests.ocr_data_utils import drop_record_fields, read_ocr_data


def test_validators() -> None:
    with pytest.raises(TypeError):
        EasyOCR(lang=12)  # ty:ignore[invalid-argument-type]


def test_easyocr_document() -> None:
    instance = EasyOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = read_ocr_data("test_data/ocr.csv")

    assert result is not None
    assert drop_record_fields(result.records, {"confidence"}) == drop_record_fields(
        expected.records, {"confidence"}
    )
