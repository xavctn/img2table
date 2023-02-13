# coding: utf-8

from img2table.ocr.tesseract import TesseractOCR

try:
    from img2table.ocr.aws_textract import TextractOCR
except ImportError:
    pass

try:
    from img2table.ocr.azure import AzureOCR
except ImportError:
    pass

try:
    from img2table.ocr.google_vision import VisionOCR
except ImportError:
    pass

try:
    from img2table.ocr.paddle import PaddleOCR
except ImportError:
    pass
