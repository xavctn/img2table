from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, overload

import numpy as np  # noqa: TC002

from img2table._validation import (
    ValidationError,
    validate_bool,
    validate_pages,
    validate_positive_int,
    validate_src,
)

if TYPE_CHECKING:
    from img2table.ocr._types import OCRData, OCRInstance
    from img2table.tables.extraction import ExtractedTable
    from img2table.tables.types import Table


@dataclass
class MockDocument:
    images: list[np.ndarray]


@dataclass
class Document:
    src: str | Path | io.BytesIO | bytes
    detect_rotation: bool = False
    pages: list[int] | None = None
    ocr_data: OCRData | None = None

    def __post_init__(self) -> None:
        validate_src(self.src)
        validate_bool(self.detect_rotation, "detect_rotation")
        validate_pages(self.pages)
        if self.pages is not None:
            self.pages = sorted(self.pages)

    @property
    def images(self) -> list[np.ndarray]:
        raise NotImplementedError

    @cached_property
    def file_bytes(self) -> bytes:
        if isinstance(self.src, bytes):
            return self.src
        if isinstance(self.src, io.BytesIO):
            self.src.seek(0)
            return self.src.read()
        if isinstance(self.src, str):
            with Path(self.src).open("rb") as f:
                return f.read()
        if isinstance(self.src, Path):
            with self.src.open("rb") as f:
                return f.read()
        raise ValidationError(f"Unsupported source type: {type(self.src)}")

    def get_table_content(
        self, tables: dict[int, list[Table]], ocr: OCRInstance | None, min_confidence: int
    ) -> dict[int, list[ExtractedTable]]:
        """
        Retrieve table content with OCR
        :param tables: dictionary containing extracted tables by page
        :param ocr: OCRInstance object used to extract table content
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: dictionary with page number as key and list of extracted tables as values
        """
        # Get pages where tables have been detected
        table_pages = [k for k, v in tables.items() if len(v) > 0]

        image_shapes = {page: self.images[page].shape[:2] for page in table_pages}

        if (self.ocr_data is None and ocr is None) or len(table_pages) == 0:
            return {
                k: [
                    tb.extracted_table(
                        image_width=image_shapes[k][1], image_height=image_shapes[k][0]
                    )
                    for tb in v
                ]
                if k in image_shapes
                else []
                for k, v in tables.items()
            }

        # Create document containing only pages with tables
        ocr_doc = MockDocument(images=[self.images[page] for page in table_pages])

        # Get OCRData object
        if self.ocr_data is None and ocr is not None:
            self.ocr_data = ocr.of(document=ocr_doc)

        if self.ocr_data is None:
            return {k: [] for k in tables}

        # Retrieve table contents with ocr
        for idx, page in enumerate(table_pages):
            # Get table content
            tables[page] = [
                table.get_content(
                    ocr_data=self.ocr_data, min_confidence=min_confidence, page_number=idx
                )
                for table in tables[page]
            ]

            # Filter relevant tables
            tables[page] = [
                table for table in tables[page] if max(table.nb_rows, table.nb_columns) >= 2
            ]

            # Retrieve titles
            for table in tables[page]:
                if table.title_area is not None:
                    table.set_title(
                        title=self.ocr_data.get_text_cell(
                            cell=table.title_area,
                            margin=5,
                            page_number=idx,
                            min_confidence=min_confidence,
                        )
                    )

        # Reset OCR
        self.ocr_data = None

        return {
            k: [
                tb.extracted_table(image_width=image_shapes[k][1], image_height=image_shapes[k][0])
                for tb in v
                if tb.has_valid_shape()
            ]
            for k, v in tables.items()
        }

    def extract_tables(
        self,
        ocr: OCRInstance | None = None,
        implicit_rows: bool = False,
        implicit_columns: bool = False,
        borderless_tables: bool = False,
        min_confidence: int = 50,
        max_workers: int = 1,
    ) -> dict[int, list[ExtractedTable]]:
        """
        Extract tables from document
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :param max_workers: number of concurrent workers used for table extraction
        :return: dictionary with page number as key and list of extracted tables as values
        """
        # Arguments validation
        validate_bool(implicit_rows, "implicit_rows")
        validate_bool(implicit_columns, "implicit_columns")
        validate_bool(borderless_tables, "borderless_tables")
        validate_positive_int(max_workers, "max_workers")
        validate_positive_int(min_confidence, "min_confidence")

        # Extract tables from document
        from img2table.tables.extractor import TableExtractor

        tables = {
            idx: TableExtractor(img=img).extract_tables(
                implicit_rows=implicit_rows,
                implicit_columns=implicit_columns,
                borderless_tables=borderless_tables,
            )
            for idx, img in enumerate(self.images)
        }

        # Update table content with OCR if possible
        tables = self.get_table_content(tables=tables, ocr=ocr, min_confidence=min_confidence)

        # If pages have been defined, modify tables keys
        if self.pages:
            return {self.pages[k]: v for k, v in tables.items()}

        return tables

    @overload
    def to_xlsx(
        self,
        dest: str | Path,
        ocr: OCRInstance | None = None,
        implicit_rows: bool = False,
        implicit_columns: bool = False,
        borderless_tables: bool = False,
        min_confidence: int = 50,
    ) -> None: ...
    @overload
    def to_xlsx(
        self,
        dest: io.BytesIO,
        ocr: OCRInstance | None = None,
        implicit_rows: bool = False,
        implicit_columns: bool = False,
        borderless_tables: bool = False,
        min_confidence: int = 50,
    ) -> io.BytesIO: ...

    def to_xlsx(
        self,
        dest: str | Path | io.BytesIO,
        ocr: OCRInstance | None = None,
        implicit_rows: bool = False,
        implicit_columns: bool = False,
        borderless_tables: bool = False,
        min_confidence: int = 50,
        max_workers: int = 1,
    ) -> io.BytesIO | None:
        """
        Create xlsx file containing all extracted tables from document
        :param dest: destination for xlsx file
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :param max_workers: number of concurrent workers used for table extraction
        :return: if a buffer is passed as dest arg, it is returned containing xlsx data
        """
        import xlsxwriter

        # Extract tables
        extracted_tables = self.extract_tables(
            ocr=ocr,
            implicit_rows=implicit_rows,
            implicit_columns=implicit_columns,
            borderless_tables=borderless_tables,
            min_confidence=min_confidence,
            max_workers=max_workers,
        )
        extracted_tables = (
            {0: extracted_tables} if isinstance(extracted_tables, list) else extracted_tables
        )

        # Create workbook
        workbook = xlsxwriter.Workbook(dest, {"in_memory": True})

        # Create generic cell format
        cell_format = workbook.add_format(
            {"align": "center", "valign": "vcenter", "text_wrap": True}
        )
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

        return None
