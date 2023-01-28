# coding: utf-8
import json
from collections import OrderedDict
from io import BytesIO

import pytest

from img2table.document.pdf import PDF
from img2table.ocr import TesseractOCR
from img2table.tables.objects.extraction import ExtractedTable, BBox, TableCell


def test_validators():
    with pytest.raises(TypeError) as e_info:
        pdf = PDF(src=1)

    with pytest.raises(TypeError) as e_info:
        pdf = PDF(src="img", dpi="8")

    with pytest.raises(TypeError) as e_info:
        pdf = PDF(src="img", dpi=200, pages=12)

    with pytest.raises(TypeError) as e_info:
        pdf = PDF(src="img", dpi=200, pages=["12"])


def test_load_pdf():
    # Load from path
    pdf_from_path = PDF(src="test_data/test.pdf", dpi=300)

    # Load from bytes
    with open("test_data/test.pdf", "rb") as f:
        pdf_from_bytes = PDF(src=f.read(), dpi=300)

    # Load from BytesIO
    with open("test_data/test.pdf", "rb") as f:
        pdf_from_bytesio = PDF(src=BytesIO(f.read()), dpi=300)

    assert pdf_from_path.bytes == pdf_from_bytes.bytes == pdf_from_bytesio.bytes

    assert list(pdf_from_path.images)[0].shape == (3300, 2550)


def test_pdf_pages():
    assert len(list(PDF(src="test_data/test.pdf").images)) == 2
    assert len(list(PDF(src="test_data/test.pdf", pages=[0]).images)) == 1


def test_pdf_tables():
    ocr = TesseractOCR()
    pdf = PDF(src="test_data/test.pdf")

    result = pdf.extract_tables(ocr=ocr, implicit_rows=True, min_confidence=50)

    with open("test_data/extracted_tables.json", "r") as f:
        expected = {
            int(k): [ExtractedTable(title=tb.get('title'),
                                    bbox=BBox(**tb.get('bbox')),
                                    content=OrderedDict({int(id): [TableCell(bbox=BBox(**c.get('bbox')),
                                                                             value=c.get('value'))
                                                                   for c in row]
                                                         for id, row in tb.get('content').items()})
                                    )
                     for tb in list_tbs]
            for k, list_tbs in json.load(f).items()
        }

    assert result == expected
