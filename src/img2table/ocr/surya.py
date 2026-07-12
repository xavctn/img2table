from __future__ import annotations

from typing import TYPE_CHECKING

from img2table.ocr._types import OCRData, OCRInstance

if TYPE_CHECKING:
    from img2table.document._types import Document, MockDocument


class SuryaOCR(OCRInstance):
    """
    Surya OCR instance
    """

    def __init__(self, langs: list[str] | None = None) -> None:
        """
        Initialization of SuryaOCR instance
        """
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor

        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Missing dependencies, please install 'img2table[surya]' to use this class."
            ) from err

        langs = ["en"] if langs is None else langs
        if isinstance(langs, list):
            if all(isinstance(lng, str) for lng in langs or []):
                self.langs = langs or ["en"]
            else:
                raise TypeError("All values should be strings for langs argument")
        else:
            raise TypeError(f"Invalid type {type(langs)} for langs argument")

        self.rec_predictor = RecognitionPredictor(SuryaInferenceManager())

    def of(self, document: Document | MockDocument) -> OCRData | None:
        """
        Convert docTR Document object to OCRData object
        :param content: docTR Document object
        :return: OCRData object corresponding to content
        """
        from bs4 import BeautifulSoup
        from PIL import Image

        # Get OCR of all images
        content = self.rec_predictor(images=[Image.fromarray(img) for img in document.images])

        # Create dict of elements by page
        records = {}

        for page_id, ocr_result in enumerate(content):
            for idx, block in enumerate(ocr_result.blocks):
                value = BeautifulSoup(block.html, features="html.parser").get_text(" ", strip=True)
                dict_word = {
                    "id": f"word_{page_id + 1}_{idx + 1}_0",
                    "parent": f"word_{page_id + 1}_{idx + 1}",
                    "value": value or None,
                    "confidence": round(100 * (block.confidence or 0)),
                    "x1": int(block.bbox[0]),
                    "y1": int(block.bbox[1]),
                    "x2": int(block.bbox[2]),
                    "y2": int(block.bbox[3]),
                }

                records.setdefault(page_id, []).append(dict_word)

        return OCRData(records=records) if records else None
