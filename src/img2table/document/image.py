import typing
from functools import cached_property
from typing import Optional

import cv2
import numpy as np

from img2table.document.base import Document
from img2table.document.base.rotation import fix_rotation_image
from img2table.tables.objects.extraction import ExtractedTable

if typing.TYPE_CHECKING:
    from img2table.ocr.base import OCRInstance


class Image(Document):
    def __model_post_init__(self) -> None:
        super().__model_post_init__()
        self.pages = [0]

    @cached_property
    def images(self) -> list[np.ndarray]:
        img = cv2.imdecode(
            buf=np.frombuffer(buffer=self.bytes, dtype=np.uint8),
            flags=cv2.IMREAD_COLOR,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.detect_rotation:
            rotated_img, _ = fix_rotation_image(img=img)
            return [rotated_img]
        return [img]

    def extract_tables(
        self,
        ocr: Optional["OCRInstance"] = None,
        implicit_rows: bool = False,
        implicit_columns: bool = False,
        borderless_tables: bool = False,
        min_confidence: int = 50,
    ) -> list[ExtractedTable]:  # ty:ignore[invalid-method-override]
        """
        Extract tables from document
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: list of extracted tables
        """
        extracted_tables = super().extract_tables(
            ocr=ocr,
            implicit_rows=implicit_rows,
            implicit_columns=implicit_columns,
            borderless_tables=borderless_tables,
            min_confidence=min_confidence,
        )

        return extracted_tables.get(0, [])
