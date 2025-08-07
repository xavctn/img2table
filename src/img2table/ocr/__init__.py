
from img2table.ocr.aws_textract import TextractOCR
from img2table.ocr.azure import AzureOCR
from img2table.ocr.doctr import DocTR
from img2table.ocr.easyocr import EasyOCR
from img2table.ocr.google_vision import VisionOCR
from img2table.ocr.paddle import PaddleOCR
from img2table.ocr.surya import SuryaOCR
from img2table.ocr.tesseract import TesseractOCR


__all__ = [
           "AzureOCR",
           "DocTR",
           "EasyOCR",
           "PaddleOCR",
           "SuryaOCR",
           "TesseractOCR",
           "TextractOCR",
           "VisionOCR",
]
