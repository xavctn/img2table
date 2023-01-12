# coding: utf-8
from typing import Any

from img2table.document.base import Document
from img2table.ocr.data import OCRDataframe


class OCRInstance:
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
