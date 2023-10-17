# coding: utf-8
import io
import typing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Union, Dict, List, Optional

import numpy as np
import xlsxwriter

from img2table import Validations
from img2table.tables.objects.extraction import ExtractedTable

if typing.TYPE_CHECKING:
    from img2table.ocr.base import OCRInstance
    from img2table.tables.objects.table import Table


@dataclass
class MockDocument:
    images: List[np.ndarray]


@dataclass
class Document(Validations):
    src: Union[str, Path, io.BytesIO, bytes]

    def validate_src(self, value, **_) -> Union[str, Path, io.BytesIO, bytes]:
        if not isinstance(value, (str, Path, io.BytesIO, bytes)):
            raise TypeError(f"Invalid type {type(value)} for src argument")
        return value

    def validate_detect_rotation(self, value, **_) -> int:
        if not isinstance(value, bool):
            raise TypeError(f"Invalid type {type(value)} for detect_rotation argument")
        return value

    def __post_init__(self):
        super(Document, self).__post_init__()
        # Initialize ocr_df
        self.ocr_df = None

        if not hasattr(self, "pages"):
            self.pages = None

        if isinstance(self.pages, list):
            self.pages = sorted(self.pages)

    @cached_property
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
    def images(self) -> List[np.ndarray]:
        raise NotImplementedError

    def get_table_content(self, tables: Dict[int, List["Table"]], ocr: "OCRInstance",
                          min_confidence: int) -> Dict[int, List[ExtractedTable]]:
        """
        Retrieve table content with OCR
        :param tables: dictionary containing extracted tables by page
        :param ocr: OCRInstance object used to extract table content
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: dictionary with page number as key and list of extracted tables as values
        """
        # Get pages where tables have been detected
        table_pages = [k for k, v in tables.items() if len(v) > 0]

        if (self.ocr_df is None and ocr is None) or len(table_pages) == 0:
            return {k: [tb.extracted_table for tb in v] for k, v in tables.items()}

        # Create document containing only pages
        ocr_doc = MockDocument(images=[self.images[page] for page in table_pages])

        # Get OCRDataFrame object
        if self.ocr_df is None and ocr is not None:
            self.ocr_df = ocr.of(document=ocr_doc)

        # Retrieve table contents with ocr
        for idx, page in enumerate(table_pages):
            ocr_df_page = self.ocr_df.page(page_number=idx)
            # Get table content
            tables[page] = [table.get_content(ocr_df=ocr_df_page, min_confidence=min_confidence)
                            for table in tables[page]]

            # Filter relevant tables
            tables[page] = [table for table in tables[page] if max(table.nb_rows, table.nb_columns) >= 2]

            # Retrieve titles
            from img2table.tables.processing.text.titles import get_title_tables
            tables[page] = get_title_tables(img=self.images[page],
                                            tables=tables[page],
                                            ocr_df=ocr_df_page)

        # Reset OCR
        self.ocr_df = None

        return {k: [tb.extracted_table for tb in v] for k, v in tables.items()}

    def extract_tables(self, ocr: "OCRInstance" = None, implicit_rows: bool = False, borderless_tables: bool = False,
                       min_confidence: int = 50) -> Dict[int, List[ExtractedTable]]:
        """
        Extract tables from document
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: dictionary with page number as key and list of extracted tables as values
        """
        # Extract tables from document
        from img2table.tables.image import TableImage
        tables = {idx: TableImage(img=img,
                                  min_confidence=min_confidence).extract_tables(implicit_rows=implicit_rows,
                                                                                borderless_tables=borderless_tables)
                  for idx, img in enumerate(self.images)}

        # Update table content with OCR if possible
        tables = self.get_table_content(tables=tables,
                                        ocr=ocr,
                                        min_confidence=min_confidence)

        # If pages have been defined, modify tables keys
        if self.pages:
            tables = {self.pages[k]: v for k, v in tables.items()}

        return tables

    def to_xlsx(self, dest: Union[str, Path, io.BytesIO], ocr: "OCRInstance" = None, implicit_rows: bool = False,
                borderless_tables: bool = False, min_confidence: int = 50) -> Optional[io.BytesIO]:
        """
        Create xlsx file containing all extracted tables from document
        :param dest: destination for xlsx file
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: if a buffer is passed as dest arg, it is returned containing xlsx data
        """
        # Extract tables
        extracted_tables = self.extract_tables(ocr=ocr,
                                               implicit_rows=implicit_rows,
                                               borderless_tables=borderless_tables,
                                               min_confidence=min_confidence)
        extracted_tables = {0: extracted_tables} if isinstance(extracted_tables, list) else extracted_tables

        # Create workbook
        workbook = xlsxwriter.Workbook(dest, {'in_memory': True})

        # Create generic cell format
        cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
        cell_format.set_border()

        # For each extracted table, create a corresponding worksheet and populate it
        for page, tables in extracted_tables.items():
            for idx, table in enumerate(tables):
                # Create worksheet
                sheet = workbook.add_worksheet(name=f"Page {page + 1} - Table {idx + 1}")

                # Populate worksheet
                table._to_worksheet(sheet=sheet, cell_fmt=cell_format)

        # Close workbook
        workbook.close()

        # If destination is a BytesIO object, return it
        if isinstance(dest, io.BytesIO):
            dest.seek(0)
            return dest
