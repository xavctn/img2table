from img2table.document.image import Image
from img2table.ocr import DocTR
from tests.ocr_data_utils import drop_record_fields, read_ocr_data


def test_doctr_document() -> None:
    instance = DocTR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = read_ocr_data("test_data/ocr.csv")

    assert result is not None
    assert drop_record_fields(result.records, {"confidence"}) == drop_record_fields(
        expected.records, {"confidence"}
    )
