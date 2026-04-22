import sys
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from img2table.document.pdf import PDF
from img2table.ocr import TesseractOCR
from img2table.tables.objects.extraction import BBox


def test_validators() -> None:
    with pytest.raises(ValidationError):
        PDF(src=1)  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValidationError):
        PDF(src="img", pages=12)  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValidationError):
        PDF(src="img", pages=["True"])  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValidationError):
        PDF(src="img", pages=[1], detect_rotation="a")  # ty:ignore[invalid-argument-type]


def test_load_pdf() -> None:
    # Load from path
    pdf_from_path = PDF(src="test_data/test.pdf")

    # Load from bytes
    with Path("test_data/test.pdf").open("rb") as f:
        pdf_from_bytes = PDF(src=f.read())

    # Load from BytesIO
    with Path("test_data/test.pdf").open("rb") as f:
        pdf_from_bytesio = PDF(src=BytesIO(f.read()))

    assert pdf_from_path.file_bytes == pdf_from_bytes.file_bytes == pdf_from_bytesio.file_bytes

    assert next(iter(pdf_from_path.images)).shape == (2200, 1700, 3)


def test_pdf_pages() -> None:
    assert len(list(PDF(src="test_data/test.pdf").images)) == 2
    assert len(list(PDF(src="test_data/test.pdf", pages=[0]).images)) == 1


def test_pdf_tables(mock_tesseract) -> None:  # noqa: ANN001, ARG001
    ocr = TesseractOCR()
    pdf = PDF(src="test_data/test.pdf")

    result = pdf.extract_tables(ocr=ocr, implicit_rows=True, min_confidence=50)

    assert result[0][0].title == "Example of Data Table 1"
    if sys.version_info < (3, 11):
        assert result[0][0].bbox == BBox(x1=235, y1=249, x2=1442, y2=543)
    assert (len(result[0][0].content), len(result[0][0].content[0])) == (5, 4)

    assert result[0][1].title == "Example of Data Table 2"
    if sys.version_info < (3, 11):
        assert result[0][1].bbox == BBox(x1=236, y1=672, x2=1452, y2=972)
    assert (len(result[0][1].content), len(result[0][1].content[0])) == (5, 4)

    assert result[1][0].title == "Example of Data Table 3"
    if sys.version_info < (3, 11):
        assert result[1][0].bbox == BBox(x1=235, y1=249, x2=1442, y2=543)
    assert (len(result[1][0].content), len(result[1][0].content[0])) == (5, 4)

    assert result[1][1].title == "Example of Data Table 4"
    if sys.version_info < (3, 11):
        assert result[1][1].bbox == BBox(x1=236, y1=672, x2=1452, y2=972)
    assert (len(result[1][1].content), len(result[1][1].content[0])) == (5, 4)
