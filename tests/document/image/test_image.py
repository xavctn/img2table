# coding: utf-8
import json
from collections import OrderedDict
from io import BytesIO

from img2table.document.image import Image
from img2table.ocr import TesseractOCR
from img2table.tables.objects.extraction import ExtractedTable, BBox, TableCell


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
    img = Image(src="test_data/test.png", ocr=ocr)

    result = img.extract_tables(implicit_rows=True, min_confidence=50)

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