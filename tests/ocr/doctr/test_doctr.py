from img2table.document.image import Image
from img2table.ocr import DocTR
from tests.ocr_data_utils import assert_ocr_content_close, read_ocr_data


def test_doctr_document() -> None:
    instance = DocTR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = read_ocr_data("test_data/ocr.csv")

    assert result is not None
    assert_ocr_content_close(result, expected)
