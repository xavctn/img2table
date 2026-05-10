from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pypdfium2 import PdfDocument, PdfTextPage

from img2table.document._types import Document, MockDocument
from img2table.ocr._types import OCRData, OCRInstance


@dataclass
class Char:
    value: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return (self.x2 - self.x1) or 1

    @property
    def height(self) -> int:
        return (self.y2 - self.y1) or 1

    def distance(self, char: Char) -> float:
        return (
            ((self.x2 + self.x1 - char.x2 - char.x1) / 2) ** 2
            + ((self.y2 + self.y1 - char.y2 - char.y1) / 2) ** 2
        ) ** 0.5


@dataclass(slots=True)
class Word:
    idx: int
    line_idx: int
    chars: list[Char] = field(default_factory=list)
    x1: int = 999999
    y1: int = 999999
    x2: int = 0
    y2: int = 0
    direction: str = "unknown"
    sum_size: float = 0.0

    def add_char(self, char: Char) -> None:
        self.chars.append(char)
        self.x1, self.y1 = min(self.x1, char.x1), min(self.y1, char.y1)
        self.x2, self.y2 = max(self.x2, char.x2), max(self.y2, char.y2)

        # Update direction
        w, h = (self.x2 - self.x1) or 1, (self.y2 - self.y1) or 1
        if len(self.chars) >= 3:
            if w / h >= 2:
                self.direction = "horizontal"
            elif h / w >= 2:
                self.direction = "vertical"

        # Update size for distance thresholding
        if self.direction == "horizontal":
            self.sum_size += char.width
        elif self.direction == "vertical":
            self.sum_size += char.height
        else:
            self.sum_size += max(char.width, char.height)

    @property
    def avg_size(self) -> float:
        return self.sum_size / len(self.chars) if self.chars else 0

    def distance(self, char: Char) -> float:
        if not self.chars:
            return 0.0
        last = self.chars[-1]
        return (
            ((last.x1 + last.x2 - char.x1 - char.x2) / 2) ** 2
            + ((last.y1 + last.y2 - char.y1 - char.y2) / 2) ** 2
        ) ** 0.5

    def corresponds(self, char: Char) -> bool:
        if not self.chars:
            return True
        w, h = (self.x2 - self.x1) or 1, (self.y2 - self.y1) or 1

        if self.direction == "horizontal":
            overlap = min(self.y2, char.y2) - max(self.y1, char.y1)
            return overlap >= 0.5 * min(h, char.height)
        if self.direction == "vertical":
            overlap = min(self.x2, char.x2) - max(self.x1, char.x1)
            return overlap >= 0.5 * min(w, char.width)

        return self.distance(char) <= 3 * self.avg_size

    def asdict(self, page_idx: int) -> dict:
        return {
            "id": f"word_{page_idx + 1}_{self.line_idx}_{self.idx}",
            "parent": f"line_{page_idx + 1}_{self.line_idx}",
            "value": "".join([c.value for c in self.chars]),
            "confidence": 99,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


def get_all_char_data(
    text_page: PdfTextPage,
    n_chars: int,
    page_width: float,
    page_height: float,
    page_rotation: int,
    x_offset: float,
    y_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute character coordinates within page
    :param text_page: PdfTextPage object from pypdfium2
    :param n_chars: number of characters in text page
    :param page_width: page width
    :param page_height: page height
    :param page_rotation: page rotation angle
    :param x_offset: page cropbox horizontal offset
    :param y_offset: page vertical horizontal offset
    :return: tuple of character coordinates within page
    """
    # Batch extract boxes
    boxes = np.array([text_page.get_charbox(i, loose=True) for i in range(n_chars)])

    # Check for empty boxes (loose vs strict)
    mask = (boxes[:, 0] == boxes[:, 2]) & (boxes[:, 1] == boxes[:, 3])
    if np.any(mask):
        for i in np.where(mask)[0]:
            boxes[i] = text_page.get_charbox(int(i), loose=False)

    _x1, _y1, _x2, _y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Rotation logic
    if page_rotation == 90:
        _x1, _y1, _x2, _y2 = _y1, page_height - _x2, _y2, page_height - _x1
    elif page_rotation == 180:
        _x1, _y1, _x2, _y2 = (
            page_width - _x1,
            page_height - _y2,
            page_width - _x2,
            page_height - _y1,
        )
    elif page_rotation == 270:
        _x1, _y1, _x2, _y2 = page_height - _y2, _x1, page_height - _y1, _x2

    # Scale and finalize
    scale = 200 / 72
    tx1 = ((np.minimum(_x1, _x2) - x_offset) * scale).astype(int)
    ty1 = ((page_height - np.maximum(_y1, _y2) + y_offset) * scale).astype(int)
    tx2 = ((np.maximum(_x1, _x2) - x_offset) * scale).astype(int)
    ty2 = ((page_height - np.minimum(_y1, _y2) + y_offset) * scale).astype(int)

    return tx1, ty1, tx2, ty2


class PdfOCR(OCRInstance):
    def of(self, document: Document | MockDocument) -> OCRData | None:
        list_pages = []

        if isinstance(document, MockDocument) or not document.pages:
            return None

        doc = PdfDocument(input=document.file_bytes)
        for idx, page_number in enumerate(document.pages):
            # Get page
            page = doc.get_page(index=page_number)

            # Get page characteristics
            page_height, page_width, page_rotation = (
                page.get_height(),
                page.get_width(),
                page.get_rotation(),
            )
            x_offset, y_offset, _, _ = page.get_cropbox()

            # Get text page
            text_page = page.get_textpage()
            char_count = text_page.count_chars()

            if char_count <= 1:
                list_pages.append([])
                continue

            # Get characters coordinates
            tx1, ty1, tx2, ty2 = get_all_char_data(
                text_page=text_page,
                n_chars=char_count,
                page_width=page_width,
                page_height=page_height,
                page_rotation=page_rotation,
                x_offset=x_offset,
                y_offset=y_offset,
            )
            text = text_page.get_text_range(index=0, count=char_count)

            word_id, line_id = 1, 1
            current_word = Word(idx=1, line_idx=1)
            words = [current_word]

            for val, x1, y1, x2, y2 in zip(text, tx1, ty1, tx2, ty2, strict=True):
                char = Char(value=val, x1=x1, y1=y1, x2=x2, y2=y2)

                if char.value.strip() == "":
                    word_id += 1
                    continue

                # Logic to decide if char belongs to the current word
                if current_word.corresponds(char):
                    if (
                        current_word.distance(char) <= 2 * current_word.avg_size
                        and word_id == current_word.idx
                    ):
                        current_word.add_char(char)
                    else:
                        word_id += 1
                        current_word = Word(idx=word_id, line_idx=line_id)
                        current_word.add_char(char)
                        words.append(current_word)
                else:
                    word_id += 1
                    line_id += 1
                    current_word = Word(idx=word_id, line_idx=line_id)
                    current_word.add_char(char)
                    words.append(current_word)

            # Filtering out words that ended up empty
            list_words = [w.asdict(page_idx=idx) for w in words if w.chars]

            if list_words:
                # Append to list of pages
                list_pages.append(list_words)
            elif len([obj for obj in page.get_objects() if obj.type == 3]) == 0:
                # Check if page is blank
                page_item = {
                    "class": "ocr_page",
                    "id": f"page_{idx + 1}",
                    "parent": None,
                    "value": None,
                    "confidence": None,
                    "x1": 0,
                    "y1": 0,
                    "x2": int(page_width * 200 / 72),
                    "y2": int(page_height * 200 / 72),
                }
                list_pages.append([page_item])
            else:
                list_pages.append([])

        doc.close()

        # Create OCRData
        records = {
            page: page_elements for page, page_elements in enumerate(list_pages) if page_elements
        }

        return OCRData(records=records) if records else None
