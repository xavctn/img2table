# coding: utf-8
import os

import cv2
import pandas as pd

from img2table.document.image import Image
from img2table.ocr import TesseractOCR
from img2table.ocr.data import OCRDataframe
from tests import MOCK_DIR


def test_tesseract_hocr(mock_tesseract):
    instance = TesseractOCR()
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    result = instance.hocr(image=img)

    with open(os.path.join(MOCK_DIR, "tesseract_hocr.html"), 'r') as f:
        assert result == f.read()


def test_tesseract_content(mock_tesseract):
    instance = TesseractOCR()
    doc = Image(src="test_data/test.png")

    result = instance.content(document=doc)

    with open(os.path.join(MOCK_DIR, "tesseract_hocr.html"), 'r') as f:
        assert list(result) == [f.read()]


def test_tesseract_ocr_df():
    instance = TesseractOCR()

    with open(os.path.join(MOCK_DIR, "tesseract_hocr.html"), 'r') as f:
        content = [f.read()]

    result = instance.to_ocr_dataframe(content=content)

    expected = OCRDataframe(df=pd.read_csv("test_data/ocr_df.csv", sep=";"))

    assert result == expected


def test_tesseract_document(mock_tesseract):
    instance = TesseractOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = OCRDataframe(df=pd.read_csv("test_data/ocr_df.csv", sep=";"))

    assert result == expected