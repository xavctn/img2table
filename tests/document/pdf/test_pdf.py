# coding: utf-8
import sys
from io import BytesIO

import pytest

from img2table.document.pdf import PDF
from img2table.ocr import TesseractOCR
from img2table.tables.objects.extraction import BBox


def test_validators():
    with pytest.raises(TypeError) as e_info:
        pdf = PDF(src=1)

    with pytest.raises(TypeError) as e_info:
        pdf = PDF(src="img", pages=12)

    with pytest.raises(TypeError) as e_info:
        pdf = PDF(src="img", pages=["12"])


def test_load_pdf():
    # Load from path
    pdf_from_path = PDF(src="test_data/test.pdf")

    # Load from bytes
    with open("test_data/test.pdf", "rb") as f:
        pdf_from_bytes = PDF(src=f.read())

    # Load from BytesIO
    with open("test_data/test.pdf", "rb") as f:
        pdf_from_bytesio = PDF(src=BytesIO(f.read()))

    assert pdf_from_path.bytes == pdf_from_bytes.bytes == pdf_from_bytesio.bytes

    assert list(pdf_from_path.images)[0].shape == (2200, 1700)


def test_pdf_pages():
    assert len(list(PDF(src="test_data/test.pdf").images)) == 2
    assert len(list(PDF(src="test_data/test.pdf", pages=[0]).images)) == 1


def test_pdf_tables(mock_tesseract):
    ocr = TesseractOCR()
    pdf = PDF(src="test_data/test.pdf")

    result = pdf.extract_tables(ocr=ocr, implicit_rows=True, min_confidence=50)

    assert result[0][0].title == "Example of Data Table 1"
    if sys.version_info.minor < 11:
        assert result[0][0].bbox == BBox(x1=236, y1=249, x2=1442, y2=543)
    assert (len(result[0][0].content), len(result[0][0].content[0])) == (5, 4)

    assert result[0][1].title == "Example of Data Table 2"
    if sys.version_info.minor < 11:
        assert result[0][1].bbox == BBox(x1=235, y1=671, x2=1451, y2=971)
    assert (len(result[0][1].content), len(result[0][1].content[0])) == (5, 4)

    assert result[1][0].title == "Example of Data Table 3"
    if sys.version_info.minor < 11:
        assert result[1][0].bbox == BBox(x1=236, y1=249, x2=1442, y2=543)
    assert (len(result[1][0].content), len(result[1][0].content[0])) == (5, 4)

    assert result[1][1].title == "Example of Data Table 4"
    if sys.version_info.minor < 11:
        assert result[1][1].bbox == BBox(x1=235, y1=671, x2=1451, y2=971)
    assert (len(result[1][1].content), len(result[1][1].content[0])) == (5, 4)
