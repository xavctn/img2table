# coding: utf-8
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import polars as pl
from pypdfium2 import PdfDocument, PdfTextPage

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe


@dataclass
class Char:
    value: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def distance(self, char: "Char") -> float:
        return (((self.x2 + self.x1 - char.x2 - char.x1) / 2) ** 2 + (
                    (self.y2 + self.y1 - char.y2 - char.y1) / 2) ** 2) ** 0.5


@dataclass
class Word:
    idx: int
    line_idx: int
    chars: List[Char]

    @property
    def x1(self) -> int:
        return min([c.x1 for c in self.chars]) if self.chars else 0

    @property
    def y1(self) -> int:
        return min([c.y1 for c in self.chars]) if self.chars else 0

    @property
    def x2(self) -> int:
        return max([c.x2 for c in self.chars]) if self.chars else 0

    @property
    def y2(self) -> int:
        return max([c.y2 for c in self.chars]) if self.chars else 0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def value(self) -> Optional[str]:
        return "".join([c.value for c in self.chars]) if self.chars else None

    def dict(self, page_idx: int) -> Dict:
        return {
            "page": page_idx,
            "class": "ocrx_word",
            "id": f"word_{page_idx + 1}_{self.line_idx}_{self.idx}",
            "parent": f"line_{page_idx + 1}_{self.line_idx}",
            "value": self.value,
            "confidence": 99,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        }

    @property
    def direction(self):
        if len(self.chars) >= 3:
            if self.width / self.height >= 2:
                return "horizontal"
            elif self.height / self.width >= 2:
                return "vertical"
        return "unknown"

    @property
    def size(self) -> float:
        if self.chars:
            if self.direction == "horizontal":
                return np.mean([c.width for c in self.chars])
            elif self.direction == "vertical":
                return np.mean([c.height for c in self.chars])
            else:
                return np.mean([max(c.height, c.width) for c in self.chars])
        else:
            return 0

    def distance(self, char: Char) -> float:
        if self.chars:
            return self.chars[-1].distance(char=char)
        else:
            return 0

    def corresponds(self, char: Char) -> bool:
        if self.chars:
            if self.direction == "horizontal":
                return min(self.y2, char.y2) - max(self.y1, char.y1) >= 0.5 * min(self.height, char.height)
            elif self.direction == "vertical":
                return min(self.x2, char.x2) - max(self.x1, char.x1) >= 0.5 * min(self.width, char.width)
            else:
                return self.distance(char=char) <= 3 * self.size
        return True

    def add_char(self, char: Char):
        self.chars.append(char)


def get_char_coordinates(text_page: PdfTextPage, idx_char: int, page_width: float,
                         page_height: float, page_rotation: int, x_offset: float,
                         y_offset: float) -> Tuple[int, int, int, int]:
    """
    Compute character coordinates within page
    :param text_page: PdfTextPage object from pypdfium2
    :param idx_char: index of character
    :param page_width: page width
    :param page_height: page height
    :param page_rotation: page rotation angle
    :param x_offset: page cropbox horizontal offset
    :param y_offset: page vertical horizontal offset
    :return: tuple of character coordinates within page
    """
    # Get character coordinates
    _x1, _y1, _x2, _y2 = text_page.get_charbox(index=idx_char, loose=True)
    if _x1 == _x2 and _y1 == _y2:
        _x1, _y1, _x2, _y2 = text_page.get_charbox(index=idx_char, loose=False)

    # Apply corrections on coordinates if page is rotated
    if page_rotation == 90:
        _x1, _y1, _x2, _y2 = _y1, page_height - _x2, _y2, page_height - _x1
    elif page_rotation == 180:
        _x1, _y1, _x2, _y2 = page_width - _x1, page_height - _y2, page_width - _x2, page_height - _y1
    elif page_rotation == 270:
        _x1, _y1, _x2, _y2 = page_height - _y2, _x1, page_height - _y2, _x2

    # Recompute coordinates with scale factor
    x1 = int((_x1 - x_offset) * 200 / 72)
    y1 = int((page_height - _y2 + y_offset) * 200 / 72)
    x2 = int((_x2 - x_offset) * 200 / 72)
    y2 = int((page_height - _y1 + y_offset) * 200 / 72)

    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


class PdfOCR(OCRInstance):
    def content(self, document: Document) -> List[List[Dict]]:
        list_pages = list()

        doc = PdfDocument(input=document.bytes)
        for idx, page_number in enumerate(document.pages):
            # Get page
            page = doc.get_page(index=page_number)

            # Get page characteristics
            page_height, page_width, page_rotation = page.get_height(), page.get_width(), page.get_cropbox()
            x_offset, y_offset, _, _ = page.get_cropbox()

            # Get text page
            text_page = page.get_textpage()

            # Extract words
            word_id, line_id, words = 1, 1, [Word(idx=1, line_idx=1, chars=[])]
            for idx_char in range(text_page.count_chars()):
                # Get character
                value = text_page.get_text_range(index=idx_char, count=1)
                x1, y1, x2, y2 = get_char_coordinates(text_page=text_page,
                                                      idx_char=idx_char,
                                                      page_width=page_width,
                                                      page_height=page_height,
                                                      page_rotation=page_rotation,
                                                      x_offset=x_offset,
                                                      y_offset=y_offset)
                char = Char(value=value, x1=x1, y1=y1, x2=x2, y2=y2)

                # Check coherency of character with previous characters / words
                if char.value.strip() == "":
                    word_id += 1
                else:
                    if words[-1].corresponds(char=char):
                        if words[-1].distance(char=char) <= 2 * words[-1].size and word_id == words[-1].idx:
                            words[-1].add_char(char=char)
                        else:
                            word_id += 1
                            words.append(Word(idx=word_id, line_idx=line_id, chars=[char]))
                    else:
                        word_id += 1
                        line_id += 1
                        words.append(Word(idx=word_id, line_idx=line_id, chars=[char]))

            # Get only words that hold values
            list_words = [w.dict(page_idx=idx) for w in words if w.value]

            if list_words:
                # Append to list of pages
                list_pages.append(list_words)
            elif len([obj for obj in page.get_objects() if obj.type == 3]) == 0:
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
                    "x2": int(page_width * 200 / 72),
                    "y2": int(page_height * 200 / 72)
                }
                list_pages.append([page_item])
            else:
                list_pages.append([])

        doc.close()
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
