# coding: utf-8
from typing import Iterator, Dict, List

import cv2
import fitz
import numpy as np

from img2table.document import Document
from img2table.ocr.pdf import PdfOCR
from img2table.tables.objects.extraction import ExtractedTable


class PDF(Document):
    @property
    def images(self) -> Iterator[np.ndarray]:
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        doc = fitz.Document(stream=self.bytes, filetype='pdf')
        for page_number in self.pages or range(doc.page_count):
            page = doc.load_page(page_id=page_number)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            yield cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def extract_tables(self, implicit_rows: bool = True, min_confidence: int = 50) -> Dict[int, List[ExtractedTable]]:
        """
        Extract tables from document
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param min_confidence: minimum confidence level from OCR in order to process text
        :return: dictionary with page number as key and list of extracted tables as values
        """
        # Try to get OCRDataframe from PDF
        self.ocr_df = PdfOCR().of(document=self)

        return super().extract_tables(implicit_rows=implicit_rows, min_confidence=min_confidence)


if __name__ == "__main__":
    from img2table.ocr import TesseractOCR
    path_pdf = r"C:\Users\xavca\Pictures\test.pdf"

    image = PDF(src=path_pdf, dpi=300, ocr=TesseractOCR(lang='fra+eng'))
    tables = image.extract_tables()

    print(tables)
