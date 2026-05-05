from img2table.document.image import Image
from tests.ocr_data_utils import read_ocr_data


def test_rapidocr_document() -> None:
    from img2table.ocr import RapidOCR

    instance = RapidOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = read_ocr_data("test_data/ocr.csv")

    assert result == expected
