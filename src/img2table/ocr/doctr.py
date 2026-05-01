from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.ocr._types import OCRData, OCRInstance

if TYPE_CHECKING:
    from img2table.document._types import Document, MockDocument


class DocTR(OCRInstance):
    """
    DocTR instance
    """

    def __init__(self, detect_language: bool = False, kw: dict | None = None) -> None:
        """
        Initialization of EasyOCR instance
        """
        try:
            from doctr.models import ocr_predictor
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Missing dependencies, please install doctr to use this class."
            ) from err

        # Create kwargs dict for constructor
        kw = kw or {}
        kw["detect_language"] = detect_language
        kw["pretrained"] = kw.get("pretrained") if kw.get("pretrained") is not None else True

        self.model = ocr_predictor(**kw)

    def of(self, document: Document | MockDocument) -> OCRData | None:
        """
        Convert docTR Document object to OCRData object
        :param content: docTR Document object
        :return: OCRData object corresponding to content
        """
        # Apply OCR
        content = self.model(document.images)

        # Create dict of elements by page
        records = {}

        for page_id, page in enumerate(content.pages):
            dimensions = page.dimensions
            word_id = 0
            for block in page.blocks:
                for line_id, line in enumerate(block.lines):
                    for word in line.words:
                        word_id += 1
                        dict_word = {
                            "id": f"word_{page_id + 1}_{line_id}_{word_id}",
                            "parent": f"word_{page_id + 1}_{line_id}",
                            "value": word.value,
                            "confidence": round(100 * word.confidence),
                            "x1": round(word.geometry[0][0] * dimensions[1]),
                            "y1": round(word.geometry[0][1] * dimensions[0]),
                            "x2": round(word.geometry[1][0] * dimensions[1]),
                            "y2": round(word.geometry[1][1] * dimensions[0]),
                        }

                        records.setdefault(page_id, []).append(dict_word)

        return OCRData(records=records) if records else None
