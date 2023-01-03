# coding: utf-8
import io
from dataclasses import dataclass
from typing import Union, Iterator, Dict, List

import numpy as np

from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe
from img2table.tables.image import TableImage
from img2table.tables.objects.extraction import ExtractedTable


@dataclass
class Document:
    src: Union[str, io.BytesIO, bytes]
    dpi: int = 300
    ocr: "OCRInstance" = None
    ocr_df: OCRDataframe = None

    @property
    def bytes(self) -> bytes:
        if isinstance(self.src, bytes):
            return self.src
        elif isinstance(self.src, io.BytesIO):
            return self.src.read()
        elif isinstance(self.src, str):
            with io.open(self.src, 'rb') as f:
                return f.read()

    @property
    def images(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def extract_tables(self, implicit_rows: bool = True, min_confidence: int = 50) -> Dict[int, List[ExtractedTable]]:
        """
        Extract tables from document
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param min_confidence: minimum confidence level from OCR in order to process text
        :return: dictionary with page number as key and list of extracted tables as values
        """
        # If possible, apply ocr to document
        if self.ocr_df is None and self.ocr is not None:
            self.ocr_df = self.ocr.of(document=self)

        # Extract tables from document
        tables = {idx: TableImage(img=img,
                                  dpi=self.dpi,
                                  ocr_df=self.ocr_df,
                                  min_confidence=min_confidence).extract_tables(implicit_rows=implicit_rows)
                  for idx, img in enumerate(self.images)}

        return tables
