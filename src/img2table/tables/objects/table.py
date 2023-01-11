# coding: utf-8
from collections import OrderedDict
from typing import Union, List

from img2table.tables.objects import TableObject
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
        for idx in reversed(empty_rows):
            self.items.pop(idx)

        # Check for empty columns and remove if necessary
        empty_cols = list()
        for idx in range(self.nb_columns):
            col_cells = [row.items[idx] for row in self.items]
            if all(map(lambda c: c.content is None, col_cells)):
                empty_cols.append(idx)
        for idx in reversed(empty_cols):
            for id_row in range(self.nb_rows):
                self.items[id_row].items.pop(idx)

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
                assert self.title == other.title
                return True
            except AssertionError:
                return False
        return False

