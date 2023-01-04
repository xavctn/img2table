# coding: utf-8
from typing import Iterator

import cv2
import fitz
import numpy as np

from img2table.document import Document


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


if __name__ == "__main__":
    from img2table.ocr import TesseractOCR
    path_pdf = r"C:\Users\xavca\Pictures\test.pdf"

    image = PDF(src=path_pdf, dpi=200, ocr=TesseractOCR(lang='fra+eng'), pages=[11])
    tables = image.extract_tables()

    print(tables)
