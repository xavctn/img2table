import typing
from dataclasses import dataclass
from typing import Optional, Any

import cv2
import numpy as np
import pypdfium2

from img2table.document.base import Document
from img2table.document.base.rotation import fix_rotation_image
from img2table.ocr.pdf import PdfOCR

if typing.TYPE_CHECKING:
    from img2table.ocr.base import OCRInstance
    from img2table.tables.objects.extraction import ExtractedTable
    from img2table.tables.objects.table import Table


@dataclass
class PDF(Document):
    pages: list[int] = None
    detect_rotation: bool = False
    pdf_text_extraction: bool = True
    _rotated: bool = False
    _images: list[np.ndarray] = None

    def validate_pages(self, value: Any, **_) -> Optional[list[int]]:
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"Invalid type {type(value)} for pages argument")
            if not all(isinstance(x, int) for x in value):
                raise TypeError("All values in pages argument should be integers")
        return value

    def validate_pdf_text_extraction(self, value: Any, **_) -> int:
        if not isinstance(value, bool):
            raise TypeError(f"Invalid type {type(value)} for pdf_text_extraction argument")
        return value

    def validate__rotated(self, value: Any, **_) -> int:
        return value

    def validate__images(self, value: Any, **_) -> int:
        return value

    @property
    def images(self) -> list[np.ndarray]:
        if self._images is not None:
            return self._images

        doc = pypdfium2.PdfDocument(input=self.bytes)

        # Get all images
        images = []
        for page_number in self.pages or range(len(doc)):
            page = doc[page_number]
            img = cv2.cvtColor(page.render(scale=200 / 72).to_numpy(), cv2.COLOR_BGR2RGB)
            # Handle rotation if needed
            if self.detect_rotation:
                final, self._rotated = fix_rotation_image(img=img)
            else:
                final, self._rotated = img, False
            images.append(final)

        self._images = images
        doc.close()
        return images

    def get_table_content(self, tables: dict[int, list["Table"]], ocr: "OCRInstance",
                          min_confidence: int) -> dict[int, list["ExtractedTable"]]:
        if not self._rotated and self.pdf_text_extraction:
            # Get pages where tables have been detected
            table_pages = [self.pages[k] if self.pages else k for k, v in tables.items() if len(v) > 0]
            images = [self.images[k] for k, v in tables.items() if len(v) > 0]

            if table_pages:
                # Create PDF object for OCR
                pdf_ocr = PDF(src=self.bytes,
                              pages=table_pages,
                              _images=images,
                              _rotated=self._rotated)

                # Try to get OCRDataframe from PDF
                self.ocr_df = PdfOCR().of(document=pdf_ocr)

        return super().get_table_content(tables=tables, ocr=ocr, min_confidence=min_confidence)
