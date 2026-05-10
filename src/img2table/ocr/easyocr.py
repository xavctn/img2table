from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.ocr._types import OCRData, OCRInstance

if TYPE_CHECKING:
    from img2table.document._types import Document, MockDocument


class EasyOCR(OCRInstance):
    """
    EAsyOCR instance
    """

    def __init__(self, lang: list[str] | None = None, kw: dict | None = None) -> None:
        """
        Initialization of EasyOCR instance
        :param lang: lang parameter used in EasyOCR
        :param kw: dictionary containing kwargs for EasyOCR constructor
        """
        try:
            from easyocr import Reader
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Missing dependencies, please install 'img2table[easyocr]' to use this class."
            ) from err

        lang = lang if lang is not None else ["en"]
        if isinstance(lang, list):
            if all(isinstance(lng, str) for lng in lang):
                self.lang = lang or ["en"]
            else:
                raise TypeError("All values should be strings for lang argument")
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        # Create kwargs dict for constructor
        if kw is not None and not isinstance(kw, dict):
            raise TypeError(f"Invalid type {type(kw)} for kw argument")
        kw = kw or {}
        kw["lang_list"] = self.lang
        kw["verbose"] = kw.get("verbose") or False

        self.reader = Reader(**kw)

    def of(self, document: Document | MockDocument) -> OCRData | None:
        """
        Convert hOCR HTML to OCRData object
        :param content: hOCR HTML string
        :return: OCRData object corresponding to content
        """
        # Get OCR of all images
        content = [self.reader.readtext(image) for image in document.images]

        # Create dict of elements by page
        records = {}

        for page, ocr_result in enumerate(content):
            for idx, word in enumerate(ocr_result):
                dict_word = {
                    "id": f"word_{page + 1}_{idx + 1}",
                    "parent": f"word_{page + 1}_{idx + 1}",
                    "value": word[1],
                    "confidence": round(100 * word[2]),
                    "x1": round(min([edge[0] for edge in word[0]])),
                    "y1": round(min([edge[1] for edge in word[0]])),
                    "x2": round(max([edge[0] for edge in word[0]])),
                    "y2": round(max([edge[1] for edge in word[0]])),
                }

                records.setdefault(page, []).append(dict_word)

        return OCRData(records=records) if records else None
