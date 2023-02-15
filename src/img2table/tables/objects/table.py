# coding: utf-8
from collections import OrderedDict
from typing import Union, List

import numpy as np

from img2table.tables.objects import TableObject
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.extraction import ExtractedTable, BBox
from img2table.tables.objects.row import Row


class Table(TableObject):
    def __init__(self, rows: Union[Row, List[Row]]):
        if rows is None:
            self._items = []
        elif isinstance(rows, Row):
            self._items = [rows]
        else:
            self._items = rows
        self._title = None

    @property
    def items(self) -> List[Row]:
        return self._items

    @property
    def title(self) -> str:
        return self._title

    def set_title(self, title: str):
        self._title = title

    @property
    def nb_rows(self) -> int:
        return len(self.items)

    @property
    def nb_columns(self) -> int:
        return self.items[0].nb_columns if self.items else 0

    @property
    def x1(self) -> int:
        return min(map(lambda x: x.x1, self.items))

    @property
    def x2(self) -> int:
        return max(map(lambda x: x.x2, self.items))

    @property
    def y1(self) -> int:
        return min(map(lambda x: x.y1, self.items))

    @property
    def y2(self) -> int:
        return max(map(lambda x: x.y2, self.items))

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def remove_rows(self, row_ids: List[int]):
        """
        Remove rows by ids
        :param row_ids: list of row ids to be removed
        """
        # Get remaining rows
        remaining_rows = [idx for idx in range(self.nb_rows) if idx not in row_ids]

        if len(remaining_rows) > 1:
            # Check created gaps between rows
            gaps = [(id_row, id_next) for id_row, id_next in zip(remaining_rows, remaining_rows[1:])
                    if id_next - id_row > 1]

            for id_row, id_next in gaps:
                # Normalize y value between rows
                y_gap = round((self.items[id_row].y2 + self.items[id_next].y1) / 2)

                # Put y value in both rows
                for c in self.items[id_row].items:
                    setattr(c, "y2", y_gap)
                for c in self.items[id_next].items:
                    setattr(c, "y1", y_gap)

        # Remove rows
        for idx in reversed(row_ids):
            self.items.pop(idx)

    def remove_columns(self, col_ids: List[int]):
        """
        Remove columns by ids
        :param col_ids: list of column ids to be removed
        """
        # Get remaining cols
        remaining_cols = [idx for idx in range(self.nb_columns) if idx not in col_ids]

        if len(remaining_cols) > 1:
            # Check created gaps between columns
            gaps = [(id_col, id_next) for id_col, id_next in zip(remaining_cols, remaining_cols[1:])
                    if id_next - id_col > 1]

            for id_col, id_next in gaps:
                # Normalize x value between columns
                x_gap = round(np.mean([row.items[id_col].x2 + row.items[id_next].x1 for row in self.items]) / 2)

                # Put y value in both columns
                for row in self.items:
                    setattr(row.items[id_col], "x2", x_gap)
                    setattr(row.items[id_next], "x1", x_gap)

        # Remove columns
        for idx in reversed(col_ids):
            for id_row in range(self.nb_rows):
                self.items[id_row].items.pop(idx)

    def get_content(self, ocr_df: "OCRDataframe", min_confidence: int = 50) -> "Table":
        """
        Retrieve text from OCRDataframe object and reprocess table to remove empty rows / columns
        :param ocr_df: OCRDataframe object
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: Table object with data attribute containing dataframe
        """
        # Get content for each cell
        self = ocr_df.get_text_table(table=self, min_confidence=min_confidence)

        # Check for empty rows and remove if necessary
        empty_rows = list()
        for idx, row in enumerate(self.items):
            if all(map(lambda c: c.content is None, row.items)):
                empty_rows.append(idx)
        self.remove_rows(row_ids=empty_rows)

        # Check for empty columns and remove if necessary
        empty_cols = list()
        for idx in range(self.nb_columns):
            col_cells = [row.items[idx] for row in self.items]
            if all(map(lambda c: c.content is None, col_cells)):
                empty_cols.append(idx)
        self.remove_columns(col_ids=empty_cols)

        # Check for uniqueness of content
        unique_cells = set([cell for row in self.items for cell in row.items])
        if len(unique_cells) == 1:
            self._items = [Row(cells=self.items[0].items[0])]

        return self

    @property
    def extracted_table(self) -> ExtractedTable:
        bbox = BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
        content = OrderedDict({idx: [cell.table_cell for cell in row.items] for idx, row in enumerate(self.items)})
        return ExtractedTable(bbox=bbox, title=self.title, content=content)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            try:
                assert self.items == other.items
                if self.title is not None:
                    assert self.title == other.title
                else:
                    assert other.title is None
                return True
            except AssertionError:
                return False
        return False

