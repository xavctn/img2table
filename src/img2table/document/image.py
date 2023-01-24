# coding: utf-8
from dataclasses import dataclass
from typing import Iterator, List

import cv2
import numpy as np

from img2table.document.base import Document
from img2table.document.base.rotation import fix_rotation_image
from img2table.tables.objects.extraction import ExtractedTable


@dataclass
class Image(Document):
    detect_rotation: bool = False

    def validate_detect_rotation(self, value, **_) -> int:
        if not isinstance(value, bool):
            raise TypeError(f"Invalid type {type(value)} for detect_rotation argument")
        return value

    def __post_init__(self):
        self.pages = None

        super(Image, self).__post_init__()

    @property
    def images(self) -> Iterator[np.ndarray]:
        img = cv2.imdecode(np.frombuffer(self.bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        yield fix_rotation_image(img=img) if self.detect_rotation else img

    def extract_tables(self, ocr: "OCRInstance" = None, implicit_rows: bool = True, min_confidence: int = 50) -> List[ExtractedTable]:
        """
        Extract tables from document
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: list of extracted tables
        """
        extracted_tables = super(Image, self).extract_tables(ocr=ocr,
                                                             implicit_rows=implicit_rows,
                                                             min_confidence=min_confidence)
        return extracted_tables.get(0)
