from typing import Any

import polars as pl

from img2table.document.base import Document
from img2table.ocr.data import OCRDataframe


class OCRInstance:
    @property
    def pl_schema(self) -> dict[str, Any]:
        return {
            "page": pl.Int64,
            "class": str,
            "id": str,
            "parent": str,
            "value": str,
            "confidence": pl.Int64,
            "x1": pl.Int64,
            "y1": pl.Int64,
            "x2": pl.Int64,
            "y2": pl.Int64
        }

    def content(self, document: Document) -> Any:
        raise NotImplementedError

    def to_ocr_dataframe(self, content: Any) -> OCRDataframe:
        raise NotImplementedError

    def of(self, document: Document) -> OCRDataframe:
        """
        Extract text from Document to OCRDataframe object
        :param document: Document object
        :return: OCRDataframe object
        """
        # Extract content from document
        content = self.content(document=document)

        # Create OCRDataframe from content
        return self.to_ocr_dataframe(content=content)
