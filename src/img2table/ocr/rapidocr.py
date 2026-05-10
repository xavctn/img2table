from __future__ import annotations

from typing import TYPE_CHECKING, cast

from img2table.ocr._types import OCRData, OCRInstance

if TYPE_CHECKING:
    from numpy import typing as npt
    from rapidocr.utils.output import RapidOCROutput

    from img2table.document._types import Document, MockDocument


class RapidOCR(OCRInstance):
    """
    Rapid-OCR instance
    """

    def __init__(self, params: dict | None = None) -> None:
        """
        Initialization of Paddle OCR instance
        :param params: parameters dictionary passed as config to RapidOCR constructor
        """
        try:
            from rapidocr import LangRec
            from rapidocr import RapidOCR as Ocr
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Missing dependencies, please install 'img2table[rapidocr]' to use this class."
            ) from err

        # Create kwargs dict for constructor
        params = params or {}
        if "Rec.lang_type" not in params:
            params["Rec.lang_type"] = LangRec.EN

        self.engine = Ocr(params=params)

    def of(self, document: Document | MockDocument) -> OCRData | None:
        ocrs = cast("list[RapidOCROutput]", [self.engine(img) for img in document.images])
        content = [
            {
                "texts": res.txts,
                "scores": res.scores,
                "boxes": res.boxes,
            }
            if res.txts is not None and res.scores is not None and res.boxes is not None
            else {}
            for res in ocrs
        ]

        # Create dict of elements by page
        records = {}
        for page, ocr_result in enumerate(content):
            for idx, (word, conf, bbox) in enumerate(
                zip(
                    cast("tuple[str, ...]", ocr_result["texts"]),
                    cast("tuple[float, ...]", ocr_result["scores"]),
                    cast("npt.NDArray", ocr_result["boxes"]),
                    strict=True,
                )
            ):
                dict_word = {
                    "id": f"word_{page + 1}_{idx + 1}",
                    "parent": f"word_{page + 1}_{idx + 1}",
                    "value": word,
                    "confidence": int(100 * conf),
                    "x1": int(min(bbox[:, 0])),
                    "y1": int(min(bbox[:, 1])),
                    "x2": int(max(bbox[:, 0])),
                    "y2": int(max(bbox[:, 1])),
                }

                records.setdefault(page, []).append(dict_word)

        return OCRData(records=records) if records else None
