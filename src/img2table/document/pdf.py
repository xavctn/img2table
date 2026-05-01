from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np  # noqa: TC002
import pypdfium2

from img2table._validation import validate_bool
from img2table.document.base import Document
from img2table.ocr.pdf import PdfOCR

if TYPE_CHECKING:
    from img2table.ocr.base import OCRInstance
    from img2table.tables.objects.extraction import ExtractedTable
    from img2table.tables.objects.table import Table


@dataclass
class PDF(Document):
    pdf_text_extraction: bool = True
    _rotated: bool = False
    _images: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_bool(self.pdf_text_extraction, "pdf_text_extraction")

    @property
    def images(self) -> list[np.ndarray]:
        if self._images is not None:
            return self._images

        doc = pypdfium2.PdfDocument(input=self.file_bytes)

        # Get all images
        images = []
        for page_number in self.pages or range(len(doc)):
            page = doc[page_number]
            img = cv2.cvtColor(page.render(scale=200 / 72).to_numpy(), cv2.COLOR_BGR2RGB)

            # Handle rotation if needed
            if self.detect_rotation:
                # Inline import to reduce library load time
                from img2table.document.base.rotation import fix_rotation_image

                img, rotated_img = fix_rotation_image(img=img)
                self._rotated = self._rotated or rotated_img

            images.append(img)

        self._images = images
        doc.close()
        return images

    def get_table_content(
        self, tables: dict[int, list[Table]], ocr: OCRInstance | None, min_confidence: int
    ) -> dict[int, list[ExtractedTable]]:
        if not self._rotated and self.pdf_text_extraction:
            # Get pages where tables have been detected
            table_pages = [
                self.pages[k] if self.pages else k for k, v in tables.items() if len(v) > 0
            ]

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
