from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from img2table.ocr._types import OCRData, OCRInstance

if TYPE_CHECKING:
    from img2table.document._types import Document, MockDocument


class PaddleOCR(OCRInstance):
    """
    Paddle-OCR instance
    """

    def __init__(self, lang: str = "en", kw: dict | None = None) -> None:
        """
        Initialization of Paddle OCR instance
        :param lang: lang parameter used in Paddle
        :param kw: dictionary containing kwargs for PaddleOCR constructor
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from paddleocr import PaddleOCR as Ocr
            except ModuleNotFoundError as err:
                raise ModuleNotFoundError(
                    "Missing dependencies, please install 'img2table[paddle]' to use this class."
                ) from err

        if isinstance(lang, str):
            self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        # Create kwargs dict for constructor
        kw = kw or {}
        kw["lang"] = self.lang
        kw["use_doc_unwarping"] = kw.get("use_doc_unwarping") or False

        self.ocr = Ocr(**kw)

    def of(self, document: Document | MockDocument) -> OCRData | None:
        ocrs = self.ocr.predict(input=document.images)
        content = [
            {
                "rec_texts": res["rec_texts"],
                "rec_scores": res["rec_scores"],
                "rec_boxes": [bbox.tolist() for bbox in res["rec_boxes"]],
            }
            for res in ocrs
        ]

        # Create dict of elements by page
        records = {}

        for page, ocr_result in enumerate(content):
            for idx, (word, conf, bbox) in enumerate(
                zip(
                    ocr_result["rec_texts"],
                    ocr_result["rec_scores"],
                    ocr_result["rec_boxes"],
                    strict=True,
                )
            ):
                dict_word = {
                    "id": f"word_{page + 1}_{idx + 1}",
                    "parent": f"word_{page + 1}_{idx + 1}",
                    "value": word,
                    "confidence": int(100 * conf),
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3]),
                }

                records.setdefault(page, []).append(dict_word)

        return OCRData(records=records) if records else None
