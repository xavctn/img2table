# coding: utf-8

from typing import List, Tuple, Dict

import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe


class EasyOCR(OCRInstance):
    """
    EAsyOCR instance
    """
    def __init__(self, lang: List[str] = ['en'], kw: Dict = None):
        """
        Initialization of EasyOCR instance
        :param lang: lang parameter used in EasyOCR
        :param kw: dictionary containing kwargs for EasyOCR constructor
        """
        try:
            from easyocr import Reader
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[easyocr]' to use this class.")

        if isinstance(lang, list):
            if all([isinstance(lng, str) for lng in lang]):
                self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        # Create kwargs dict for constructor
        kw = kw or {}
        kw["lang_list"] = self.lang
        kw["verbose"] = kw.get("verbose") or False

        self.reader = Reader(**kw)

    def content(self, document: Document) -> List[List[Tuple]]:
        # Get OCR of all images
        ocrs = [self.reader.readtext(image) for image in document.images]

        return ocrs

    def to_ocr_dataframe(self, content: List[List]) -> OCRDataframe:
        """
        Convert hOCR HTML to OCRDataframe object
        :param content: hOCR HTML string
        :return: OCRDataframe object corresponding to content
        """
        # Create list of elements
        list_elements = list()

        for page, ocr_result in enumerate(content):
            word_id = 0
            for word in ocr_result:
                word_id += 1
                dict_word = {
                    "page": page,
                    "class": "ocrx_word",
                    "id": f"word_{page + 1}_{word_id}",
                    "parent": f"word_{page + 1}_{word_id}",
                    "value": word[1],
                    "confidence": round(100 * word[2]),
                    "x1": round(min([edge[0] for edge in word[0]])),
                    "y1": round(min([edge[1] for edge in word[0]])),
                    "x2": round(max([edge[0] for edge in word[0]])),
                    "y2": round(max([edge[1] for edge in word[0]]))
                }

                list_elements.append(dict_word)

        return OCRDataframe(df=pl.LazyFrame(list_elements)) if list_elements else None
