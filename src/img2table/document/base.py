# coding: utf-8
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Iterator, Dict, List

import numpy as np

from img2table import Validations
from img2table.tables.objects.extraction import ExtractedTable


@dataclass
class Document(Validations):
    src: Union[str, Path, io.BytesIO, bytes]
    dpi: int = 200

    def validate_src(self, value, **_) -> Union[str, Path, io.BytesIO, bytes]:
        if not isinstance(value, (str, Path, io.BytesIO, bytes)):
            raise TypeError(f"Invalid type {type(value)} for src argument")
        return value

    def validate_dpi(self, value, **_) -> int:
        if not isinstance(value, int):
            raise TypeError(f"Invalid type {type(value)} for dpi argument")
        return value

    def __post_init__(self):
        super(Document, self).__post_init__()
        # Initialize ocr_df
        self.ocr_df = None

        if isinstance(self.pages, list):
            self.pages = sorted(self.pages)

    @property
    def bytes(self) -> bytes:
        if isinstance(self.src, bytes):
            return self.src
        elif isinstance(self.src, io.BytesIO):
            self.src.seek(0)
            return self.src.read()
        elif isinstance(self.src, str):
            with io.open(self.src, 'rb') as f:
                return f.read()

    @property
    def images(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def extract_tables(self, ocr: "OCRInstance" = None, implicit_rows: bool = True, min_confidence: int = 50) -> Dict[int, List[ExtractedTable]]:
        """
        Extract tables from document
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: dictionary with page number as key and list of extracted tables as values
        """
        # If possible, apply ocr to document
        if self.ocr_df is None and ocr is not None:
            self.ocr_df = ocr.of(document=self)

        # Extract tables from document
        from img2table.tables.image import TableImage
        tables = {idx: TableImage(img=img,
                                  dpi=self.dpi,
                                  ocr_df=self.ocr_df.page(page_number=idx) if self.ocr_df else None,
                                  min_confidence=min_confidence).extract_tables(implicit_rows=implicit_rows)
                  for idx, img in enumerate(self.images)}

        # If pages have been defined, modify tables keys
        if self.pages:
            tables = {self.pages[k]: v for k, v in tables.items()}

        # Reset ocr_df attribute
        self.ocr_df = None

        return tables
