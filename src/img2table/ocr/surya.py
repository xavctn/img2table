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
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
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

        # Initialize model
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor(FoundationPredictor())

    def of(self, document: Document | MockDocument) -> OCRData | None:
        """
        Convert docTR Document object to OCRData object
        :param content: docTR Document object
        :return: OCRData object corresponding to content
        """
        from PIL import Image

        # Get OCR of all images
        content = self.rec_predictor(
            images=[Image.fromarray(img) for img in document.images],
            langs=self.langs,  # ty:ignore[unknown-argument]
            det_predictor=self.det_predictor,
        )

        # Create dict of elements by page
        records = {}

        for page_id, ocr_result in enumerate(content):
            for idx, text_line in enumerate(ocr_result.text_lines):
                dict_word = {
                    "id": f"word_{page_id + 1}_{idx + 1}_0",
                    "parent": f"word_{page_id + 1}_{idx + 1}",
                    "value": text_line.text or None,
                    "confidence": round(100 * (text_line.confidence or 0)),
                    "x1": int(text_line.bbox[0]),
                    "y1": int(text_line.bbox[1]),
                    "x2": int(text_line.bbox[2]),
                    "y2": int(text_line.bbox[3]),
                }

                records.setdefault(page_id, []).append(dict_word)

        return OCRData(records=records) if records else None
