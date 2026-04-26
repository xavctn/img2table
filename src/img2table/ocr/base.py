from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from img2table.document.base import Document, MockDocument
    from img2table.ocr.data import OCRData


class OCRInstance:
    @property
    def ocr_fields(self) -> tuple[str, ...]:
        return ("id", "parent", "value", "confidence", "x1", "y1", "x2", "y2")

    def of(self, document: Document | MockDocument) -> OCRData | None:
        """
        Extract text from Document to OCRData object
        :param document: Document object
        :return: OCRData object
        """
        raise NotImplementedError
