from img2table.document.pdf import PDF
from img2table.ocr.pdf import PdfOCR
from tests.ocr_data_utils import assert_ocr_content_close, read_ocr_data


def test_pdf_document() -> None:
    instance = PdfOCR()
    doc = PDF(src="test_data/test.pdf", pages=[0, 1])

    result = instance.of(document=doc)

    expected = read_ocr_data("test_data/ocr.csv")

    assert result is not None
    assert_ocr_content_close(result, expected, tolerance=4, check_ids=True)
