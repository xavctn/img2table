# coding: utf-8
import json
from collections import OrderedDict
from io import BytesIO

import pytest
from openpyxl import load_workbook

from img2table.document.image import Image
from img2table.ocr import TesseractOCR
from img2table.tables.objects.extraction import ExtractedTable, BBox, TableCell


def test_validators():
    with pytest.raises(TypeError) as e_info:
        img = Image(src=1)

    with pytest.raises(TypeError) as e_info:
        img = Image(src="img", dpi="8")

    with pytest.raises(TypeError) as e_info:
        img = Image(src="img", detect_rotation=3)


def test_load_image():
    # Load from path
    img_from_path = Image(src="test_data/test.png")

    # Load from bytes
    with open("test_data/test.png", "rb") as f:
        img_from_bytes = Image(src=f.read())

    # Load from BytesIO
    with open("test_data/test.png", "rb") as f:
        img_from_bytesio = Image(src=BytesIO(f.read()))

    assert img_from_path.bytes == img_from_bytes.bytes == img_from_bytesio.bytes

    assert list(img_from_path.images)[0].shape == (417, 1365)


def test_image_tables(mock_tesseract):
    ocr = TesseractOCR()
    img = Image(src="test_data/test.png")

    result = img.extract_tables(ocr=ocr, implicit_rows=True, min_confidence=50)

    with open("test_data/extracted_tables.json", "r") as f:
        expected = [ExtractedTable(title=tb.get('title'),
                                   bbox=BBox(**tb.get('bbox')),
                                   content=OrderedDict({int(id): [TableCell(bbox=BBox(**c.get('bbox')),
                                                                            value=c.get('value'))
                                                                  for c in row]
                                                        for id, row in tb.get('content').items()})
                                   )
                    for tb in json.load(f)]

    assert result == expected


def test_image_excel(mock_tesseract):
    ocr = TesseractOCR()
    img = Image(src="test_data/test.png")

    result = img.to_xlsx(dest=BytesIO(), ocr=ocr, implicit_rows=True, min_confidence=50)

    expected = load_workbook(filename="test_data/expected.xlsx")
    result_wb = load_workbook(filename=result)

    for idx, ws in enumerate(result_wb.worksheets):
        assert ws.title == expected.worksheets[idx].title
        assert list(ws.values) == list(expected.worksheets[idx].values)
