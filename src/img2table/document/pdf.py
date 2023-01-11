# coding: utf-8
from dataclasses import dataclass
from typing import Iterator, Dict, List, Optional

import cv2
import fitz
import numpy as np

from img2table.document.base import Document
from img2table.ocr.pdf import PdfOCR
from img2table.tables.objects.extraction import ExtractedTable


@dataclass
class PDF(Document):
    pages: List[int] = None

    def validate_pages(self, value, **_) -> Optional[List[int]]:
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"Invalid type {type(value)} for pages argument")
            if not all(isinstance(x, int) for x in value):
                raise TypeError("All values in pages argument should be integers")
        return value

    @property
    def images(self) -> Iterator[np.ndarray]:
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        doc = fitz.Document(stream=self.bytes, filetype='pdf')
        for page_number in self.pages or range(doc.page_count):
            page = doc.load_page(page_id=page_number)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            yield cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def extract_tables(self, ocr: "OCRInstance" = None, implicit_rows: bool = True, min_confidence: int = 50) -> Dict[int, List[ExtractedTable]]:
        """
        Extract tables from document
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: dictionary with page number as key and list of extracted tables as values
        """
        # Try to get OCRDataframe from PDF
        self.ocr_df = PdfOCR().of(document=self)

        return super().extract_tables(ocr=ocr, implicit_rows=implicit_rows, min_confidence=min_confidence)
