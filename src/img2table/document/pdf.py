from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from threading import Lock
from typing import TYPE_CHECKING

import cv2
import numpy as np  # noqa: TC002
import pypdfium2

from img2table._validation import validate_bool, validate_positive_int
from img2table.document._types import Document
from img2table.ocr.pdf import PdfOCR
from img2table.tables.extractor import TableExtractor

if TYPE_CHECKING:
    from img2table.ocr._types import OCRInstance
    from img2table.tables.extraction import ExtractedTable
    from img2table.tables.types import Table


def _detect_tables(
    img: np.ndarray,
    implicit_rows: bool,
    implicit_columns: bool,
    borderless_tables: bool,
) -> list[Table]:
    return TableExtractor(img=img).extract_tables(
        implicit_rows=implicit_rows,
        implicit_columns=implicit_columns,
        borderless_tables=borderless_tables,
    )


def _render_and_detect_tables(
    page_number: int,
    file_bytes: bytes,
    pdfium_lock: Lock,
    detect_rotation: bool,
    implicit_rows: bool,
    implicit_columns: bool,
    borderless_tables: bool,
) -> tuple[np.ndarray, list[Table], bool]:
    with pdfium_lock:
        page_doc = None
        try:
            page_doc = pypdfium2.PdfDocument(input=file_bytes)
            page = page_doc[page_number]
            img = cv2.cvtColor(page.render(scale=200 / 72).to_numpy(), cv2.COLOR_BGR2RGB)
        finally:
            if page_doc is not None:
                page_doc.close()

    rotated_img = False
    if detect_rotation:
        from img2table.document.rotation import fix_rotation_image

        img, rotated_img = fix_rotation_image(img=img)

    tables = TableExtractor(img=img).extract_tables(
        implicit_rows=implicit_rows,
        implicit_columns=implicit_columns,
        borderless_tables=borderless_tables,
    )

    return img, tables, rotated_img


@dataclass
class PDF(Document):
    pdf_text_extraction: bool = True
    _rotated: bool = False
    _images: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        validate_bool(self.pdf_text_extraction, "pdf_text_extraction")
        super().__post_init__()

    def _ensure_pages(self) -> None:
        if self.pages is None:
            doc = None
            try:
                doc = pypdfium2.PdfDocument(input=self.file_bytes)
                self.pages = list(range(len(doc)))
            finally:
                if doc is not None:
                    doc.close()

    @property
    def images(self) -> list[np.ndarray]:
        if self._images is None:
            self._ensure_pages()

            # Assertion for type checking
            assert self.pages is not None

            doc = None
            try:
                doc = pypdfium2.PdfDocument(input=self.file_bytes)

                # Get all images
                images = []
                for page_number in self.pages:
                    page = doc[page_number]
                    img = cv2.cvtColor(page.render(scale=200 / 72).to_numpy(), cv2.COLOR_BGR2RGB)

                    # Handle rotation if needed
                    if self.detect_rotation:
                        # Inline import to reduce library load time
                        from img2table.document.rotation import fix_rotation_image

                        img, rotated_img = fix_rotation_image(img=img)
                        self._rotated = self._rotated or rotated_img

                    images.append(img)
            finally:
                if doc is not None:
                    doc.close()

            self._images = images

        return self._images

    def get_table_content(
        self, tables: dict[int, list[Table]], ocr: OCRInstance | None, min_confidence: int
    ) -> dict[int, list[ExtractedTable]]:
        self._ensure_pages()

        # Assertion for type checking
        assert self.pages is not None

        if not self._rotated and self.pdf_text_extraction:
            # Get pages where tables have been detected
            table_pages = [self.pages[k] for k, v in tables.items() if len(v) > 0]

            if table_pages:
                # Create PDF object for OCR
                pdf_ocr = PDF(
                    src=self.file_bytes,
                    pages=table_pages,
                    _images=[self.images[k] for k, v in tables.items() if len(v) > 0],
                )

                # Try to get OCRData from PDF
                self.ocr_data = PdfOCR().of(document=pdf_ocr)

        return super().get_table_content(tables=tables, ocr=ocr, min_confidence=min_confidence)

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
        Extract tables from PDF
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

        self._ensure_pages()

        # Assertion for type checking
        assert self.pages is not None

        if max_workers == 1 or len(self.pages) == 1:
            # No parallel processing needed
            return super().extract_tables(
                ocr=ocr,
                implicit_rows=implicit_rows,
                implicit_columns=implicit_columns,
                borderless_tables=borderless_tables,
                min_confidence=min_confidence,
                max_workers=max_workers,
            )

        if self._images is not None:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                tables = dict(
                    enumerate(
                        pool.map(
                            _detect_tables,
                            self.images,
                            repeat(implicit_rows),
                            repeat(implicit_columns),
                            repeat(borderless_tables),
                        )
                    )
                )
        else:
            pdfium_lock = Lock()
            self._images, tables = [], {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for page_idx, (img, page_tables, rotated) in enumerate(
                    pool.map(
                        _render_and_detect_tables,
                        self.pages,
                        repeat(self.file_bytes),
                        repeat(pdfium_lock),
                        repeat(self.detect_rotation),
                        repeat(implicit_rows),
                        repeat(implicit_columns),
                        repeat(borderless_tables),
                    )
                ):
                    self._images.append(img)
                    self._rotated = self._rotated or rotated
                    tables[page_idx] = page_tables

        tables = self.get_table_content(tables=tables, ocr=ocr, min_confidence=min_confidence)

        return {self.pages[k]: v for k, v in tables.items()}
