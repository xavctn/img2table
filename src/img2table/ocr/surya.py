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
            from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor

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
        self.det_processor, self.det_model = load_det_processor(), load_det_model()
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()

    def content(self, document: Document) -> typing.List["surya.schema.OCRResult"]:
        from surya.ocr import run_ocr
        # Get OCR of all images
        ocrs = run_ocr(images=[Image.fromarray(img) for img in document.images],
                       langs=[self.langs],
                       det_model=self.det_model,
                       det_processor=self.det_processor,
                       rec_model=self.rec_model,
                       rec_processor=self.rec_processor)

        return ocrs

    def to_ocr_dataframe(self, content: typing.List["surya.schema.OCRResult"]) -> OCRDataframe:
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
