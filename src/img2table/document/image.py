# coding: utf-8
from typing import Iterator

import cv2
import numpy as np

from img2table.document import Document


class Image(Document):
    @property
    def images(self) -> Iterator[np.ndarray]:
        yield cv2.imdecode(np.frombuffer(self.bytes, np.uint8), cv2.IMREAD_GRAYSCALE)


if __name__ == "__main__":
    from img2table.ocr import TesseractOCR
    path_img = r"C:\Users\xavca\Pictures\test_1.PNG"

    image = Image(src=path_img, dpi=200, ocr=TesseractOCR(n_threads=1, lang='fra+eng'))
    tables = image.extract_tables()

    print(tables)
