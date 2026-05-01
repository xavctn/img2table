from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from img2table._validation import validate_ocr_records

if TYPE_CHECKING:
    from img2table.document._types import Document, MockDocument
    from img2table.tables.types import Cell, Table


class OCRInstance:
    def of(self, document: Document | MockDocument) -> OCRData | None:
        """
        Extract text from Document to OCRData object
        :param document: Document object
        :return: OCRData object
        """
        raise NotImplementedError


OCRRecord = dict[str, Any]
OCRPageRecords = dict[int, list[OCRRecord]]


@dataclass
class OCRData:
    records: OCRPageRecords

    def __post_init__(self) -> None:
        validate_ocr_records(self.records, "records")

    @staticmethod
    def _group_words_by_parent(words: list[OCRRecord]) -> list[str]:
        parent_words: dict[Any, list[OCRRecord]] = defaultdict(list)
        for word in words:
            parent_words[word.get("parent")].append(word)

        lines = [
            {
                "x1": min(word["x1"] for word in words_line),
                "y1": min(word["y1"] for word in words_line),
                "value": " ".join(
                    [word["value"] for word in sorted(words_line, key=lambda wrd: wrd["x1"])]
                ).strip(),
            }
            for words_line in parent_words.values()
        ]

        return [line["value"] for line in sorted(lines, key=lambda line: (line["y1"], line["x1"]))]

    def get_text_cell(
        self,
        cell: Cell,
        margin: int = 0,
        page_number: int | None = None,
        min_confidence: int = 50,
    ) -> str | None:
        """
        Get text corresponding to cell
        :param cell: Cell object in document
        :param margin: margin to take around cell
        :param page_number: page number of the cell
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: text contained in cell
        """
        # Get cell bbox coordinates
        x1, y1, x2, y2 = cell.bbox(margin=margin)

        # Get contained words
        words_contained = [
            word
            for word in self.records.get(page_number or 0, [])
            if word["confidence"] >= min_confidence
            and (x_overlap := max(0, min(x2, word["x2"]) - max(x1, word["x1"]))) > 0
            and (y_overlap := max(0, min(y2, word["y2"]) - max(y1, word["y1"]))) > 0
            and x_overlap * y_overlap > 0.5 * (word["x2"] - word["x1"]) * (word["y2"] - word["y1"])
        ]

        text_lines = self._group_words_by_parent(words=words_contained)
        return "\n".join(text_lines).strip() or None

    def get_text_table(
        self, table: Table, page_number: int | None = None, min_confidence: int = 50
    ) -> Table:
        """
        Identify text located in Table object
        :param table: Table object
        :param page_number: page number of the cell
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: table with content set on all cells
        """
        # Get relevant words and cells
        words = [
            word
            for word in self.records.get(page_number or 0, [])
            if word["confidence"] >= min_confidence
        ]
        cells = [
            {
                "row": id_row,
                "col": id_col,
                "x1": cell.x1,
                "y1": cell.y1,
                "x2": cell.x2,
                "y2": cell.y2,
            }
            for id_row, row in enumerate(table.rows)
            for id_col, cell in enumerate(row.cells)
        ]

        if not words or not cells:
            # Impossible to map
            return table

        # Create arrays for words and cells
        words_array = np.array(
            [[word["x1"], word["y1"], word["x2"], word["y2"]] for word in words], dtype=float
        )
        cells_array = np.array(
            [[cell["x1"], cell["y1"], cell["x2"], cell["y2"]] for cell in cells], dtype=float
        )

        # Get intersection and word area for each word-cell pair
        x_left = np.maximum(words_array[:, None, 0], cells_array[None, :, 0])
        y_top = np.maximum(words_array[:, None, 1], cells_array[None, :, 1])
        x_right = np.minimum(words_array[:, None, 2], cells_array[None, :, 2])
        y_bottom = np.minimum(words_array[:, None, 3], cells_array[None, :, 3])

        intersection = np.maximum(x_right - x_left, 0) * np.maximum(y_bottom - y_top, 0)
        word_area = (words_array[:, 2] - words_array[:, 0]) * (
            words_array[:, 3] - words_array[:, 1]
        )
        mask = (
            np.divide(
                intersection,
                word_area[:, None],
                out=np.zeros_like(intersection),
                where=word_area[:, None] > 0,
            )
            > 0.5
        )

        # For each cell, group words by parent
        grouped: dict[tuple[int, int, Any], list[OCRRecord]] = defaultdict(list)
        for word_idx, cell_idx in np.argwhere(mask):
            cell = cells[int(cell_idx)]
            word = words[int(word_idx)]
            grouped[(cell["row"], cell["col"], word.get("parent"))].append(word)

        # Retrieve lines for each cell
        cell_lines: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for (row, col, _parent), words_line in grouped.items():
            cell_lines[(row, col)].append(
                {
                    "x1": min(word["x1"] for word in words_line),
                    "y1": min(word["y1"] for word in words_line),
                    "value": " ".join(
                        str(word["value"]) for word in sorted(words_line, key=lambda wrd: wrd["x1"])
                    ).strip(),
                }
            )

        # Assign text to each cell
        for (row, col), lines in cell_lines.items():
            sorted_lines = sorted(lines, key=lambda line: (line["y1"], line["x1"]))
            text = "\n".join(line["value"] for line in sorted_lines).strip()
            table.rows[row].cells[col].content = text or None

        return table

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return {
                page: sorted(records, key=lambda rec: str(rec.get("id")))
                for page, records in self.records.items()
            } == {
                page: sorted(records, key=lambda rec: str(rec.get("id")))
                for page, records in other.records.items()
            }
        return False
