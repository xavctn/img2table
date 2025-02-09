# coding: utf-8

import typing

import polars as pl
from PIL import Image

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe

if typing.TYPE_CHECKING:
    import surya


class SuryaOCR(OCRInstance):
    """
    DocTR instance
    """
    def __init__(self, langs: typing.List[str] = None):
        """
        Initialization of EasyOCR instance
        """
        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor

        except ModuleNotFoundError:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[surya]' to use this class.")

        if isinstance(langs, list):
            if all([isinstance(lng, str) for lng in langs]):
                self.langs = langs or ["en"]
            else:
                raise TypeError(f"All values should be strings for langs argument")
        else:
            raise TypeError(f"Invalid type {type(langs)} for langs argument")

        # Initialize model
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor()

    def content(self, document: Document) -> typing.List["surya.recognition.schema.OCRResult"]:
        # Get OCR of all images
        ocrs = self.rec_predictor(images=[Image.fromarray(img) for img in document.images],
                                  langs=[self.langs],
                                  det_predictor=self.det_predictor)

        return ocrs

    def to_ocr_dataframe(self, content: typing.List["surya.recognition.schema.OCRResult"]) -> OCRDataframe:
        """
        Convert docTR Document object to OCRDataframe object
        :param content: docTR Document object
        :return: OCRDataframe object corresponding to content
        """
        # Create list of elements
        list_elements = list()

        for page_id, ocr_result in enumerate(content):
            line_id = 0
            for text_line in ocr_result.text_lines:
                line_id += 1
                dict_word = {
                    "page": page_id,
                    "class": "ocrx_word",
                    "id": f"word_{page_id + 1}_{line_id}_0",
                    "parent": f"word_{page_id + 1}_{line_id}",
                    "value": text_line.text,
                    "confidence": round(100 * text_line.confidence),
                    "x1": int(text_line.bbox[0]),
                    "y1": int(text_line.bbox[1]),
                    "x2": int(text_line.bbox[2]),
                    "y2": int(text_line.bbox[3])
                }

                list_elements.append(dict_word)

        return OCRDataframe(df=pl.DataFrame(list_elements)) if list_elements else None
