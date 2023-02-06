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
        for idx, page_number in enumerate(document.pages or range(doc.page_count)):
            # Get page
            page = doc.load_page(page_id=page_number)

            # Get image size and page dimensions
            img_height, img_width = list(document.images)[idx].shape[:2]
            page_height, page_width = page.mediabox.height, page.mediabox.width

            # Extract words
            list_words = list()
            for word in page.get_text("words", sort=True):
                x1, y1, x2, y2, value, block_no, line_no, word_no = word
                dict_word = {
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
                list_words.append(dict_word)

            # Append to list of pages
            list_pages.append(list_words)

        return list_pages

    def to_ocr_dataframe(self, content: List[List[Dict]]) -> OCRDataframe:
        # Check if any page has words
        if min(map(len, content)) == 0:
            return None

        # Create OCRDataframe
        list_dfs = list(map(pl.from_dicts, content))

        return OCRDataframe(df=pl.concat(list_dfs).lazy())
