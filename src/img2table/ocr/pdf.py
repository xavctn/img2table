# coding: utf-8
from typing import List, Dict

import fitz
import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe


class PdfOCR(OCRInstance):
    def content(self, document: Document) -> List[List[Dict]]:
        list_pages = list()

        doc = fitz.Document(stream=document.bytes, filetype='pdf')
        for idx, page_number in enumerate(document.pages):
            # Get page
            page = doc.load_page(page_id=page_number)

            # Get image size and page dimensions
            img_height, img_width = list(document.images)[idx].shape[:2]
            page_height = (page.cropbox * page.rotation_matrix).height
            page_width = (page.cropbox * page.rotation_matrix).width

            # Extract words
            list_words = list()
            for x1, y1, x2, y2, value, block_no, line_no, word_no in page.get_text("words", sort=True):
                (x1, y1), (x2, y2) = fitz.Point(x1, y1) * page.rotation_matrix, fitz.Point(x2, y2) * page.rotation_matrix
                x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                word = {
                    "page": idx,
                    "class": "ocrx_word",
                    "id": f"word_{idx + 1}_{block_no}_{line_no}_{word_no}",
                    "parent": f"line_{idx + 1}_{block_no}_{line_no}",
                    "value": value,
                    "confidence": 99,
                    "x1": round(x1 * img_width / page_width),
                    "y1": round(y1 * img_height / page_height),
                    "x2": round(x2 * img_width / page_width),
                    "y2": round(y2 * img_height / page_height)
                }
                list_words.append(word)

            if list_words:
                # Append to list of pages
                list_pages.append(list_words)
            elif len(page.get_images()) == 0:
                # Check if page is blank
                page_item = {
                    "page": idx,
                    "class": "ocr_page",
                    "id": f"page_{idx + 1}",
                    "parent": None,
                    "value": None,
                    "confidence": None,
                    "x1": 0,
                    "y1": 0,
                    "x2": img_width,
                    "y2": img_height
                }
                list_pages.append([page_item])
            else:
                list_pages.append([])

        return list_pages

    def to_ocr_dataframe(self, content: List[List[Dict]]) -> OCRDataframe:
        # Check if any page has words
        if min(map(len, content)) == 0:
            return None

        # Create OCRDataframe
        list_dfs = list()
        for page_elements in content:
            if page_elements:
                list_dfs.append(pl.DataFrame(data=page_elements, schema=self.pl_schema))

        return OCRDataframe(df=pl.concat(list_dfs)) if list_dfs else None
