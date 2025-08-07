from typing import Optional

import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe


class EasyOCR(OCRInstance):
    """
    EAsyOCR instance
    """
    def __init__(self, lang: Optional[list[str]] = None, kw: Optional[dict] = None) -> None:
        """
        Initialization of EasyOCR instance
        :param lang: lang parameter used in EasyOCR
        :param kw: dictionary containing kwargs for EasyOCR constructor
        """
        try:
            from easyocr import Reader
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[easyocr]' to use this class.") from err

        lang = lang or ["en"]
        if isinstance(lang, list):
            if all(isinstance(lng, str) for lng in lang):
                self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        # Create kwargs dict for constructor
        kw = kw or {}
        kw["lang_list"] = self.lang
        kw["verbose"] = kw.get("verbose") or False

        self.reader = Reader(**kw)

    def content(self, document: Document) -> list[list[tuple]]:
        # Get OCR of all images
        return [self.reader.readtext(image) for image in document.images]

    def to_ocr_dataframe(self, content: list[list]) -> OCRDataframe:
        """
        Convert hOCR HTML to OCRDataframe object
        :param content: hOCR HTML string
        :return: OCRDataframe object corresponding to content
        """
        # Create list of elements
        list_elements = []

        for page, ocr_result in enumerate(content):
            for idx, word in enumerate(ocr_result):
                dict_word = {
                    "page": page,
                    "class": "ocrx_word",
                    "id": f"word_{page + 1}_{idx + 1}",
                    "parent": f"word_{page + 1}_{idx + 1}",
                    "value": word[1],
                    "confidence": round(100 * word[2]),
                    "x1": round(min([edge[0] for edge in word[0]])),
                    "y1": round(min([edge[1] for edge in word[0]])),
                    "x2": round(max([edge[0] for edge in word[0]])),
                    "y2": round(max([edge[1] for edge in word[0]]))
                }

                list_elements.append(dict_word)

        return OCRDataframe(df=pl.DataFrame(list_elements)) if list_elements else None
