# coding: utf-8
import copy
import math
from typing import Union, List

import numpy as np
import pandas as pd

from img2table.objects.ocr import OCRPage
from img2table.utils.data_processing import remove_empty_rows, remove_empty_columns


class TableObject(object):
    def bbox(self, margin: int = 0, height_margin: int = 0, width_margin: int = 0) -> tuple:
        """
        Return bounding box corresponding to the object
        :param margin: general margin used for the bounding box
        :param height_margin: vertical margin used for the bounding box
        :param width_margin: horizontal margin used for the bounding box
        :return: tuple representing a bounding box
        """
        # Apply margin on bbox
        if margin != 0:
            bbox = (self.x1 - margin,
                    self.y1 - margin,
                    self.x2 + margin,
                    self.y2 + margin)
        else:
            bbox = (self.x1 - width_margin,
                    self.y1 - height_margin,
                    self.x2 + width_margin,
                    self.y2 + height_margin)

        return bbox

    @property
    def upper_bound(self) -> tuple:
        return self.x1, self.y1, self.x2, self.y1

    @property
    def lower_bound(self) -> tuple:
        return self.x1, self.y2, self.x2, self.y2

    @property
    def left_bound(self) -> tuple:
        return self.x1, self.y1, self.x1, self.y2

    @property
    def right_bound(self) -> tuple:
        return self.x2, self.y1, self.x2, self.y2

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def width(self) -> int:
        return self.x2 - self.x1


class Line(TableObject):
    def __init__(self, x1: int = None, y1: int = None, x2: int = None,
                 y2: int = None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def angle(self) -> float:
        delta_x = self.x2 - self.x1
        delta_y = self.y2 - self.y1

        return math.atan2(delta_y, delta_x) * 180 / np.pi

    @property
    def length(self) -> float:
        return np.sqrt(self.height ** 2 + self.width ** 2)

    @property
    def horizontal(self) -> bool:
        return self.angle % 180 == 0

    @property
    def vertical(self) -> bool:
        return self.angle % 180 == 90

    def reprocess(self):
        # Reallocate coordinates in proper order
        _x1 = min(self.x1, self.x2)
        _x2 = max(self.x1, self.x2)
        _y1 = min(self.y1, self.y2)
        _y2 = max(self.y1, self.y2)
        self.x1, self.x2, self.y1, self.y2 = _x1, _x2, _y1, _y2

        # Correct "almost" horizontal or vertical lines
        if abs(self.angle) <= 5:
            y_val = round((self.y1 + self.y2) / 2)
            self.y2 = self.y1 = y_val
        elif abs(self.angle - 90) <= 5:
            x_val = round((self.x1 + self.x2) / 2)
            self.x2 = self.x1 = x_val

        return self


class Cell(TableObject):
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2

    @classmethod
    def from_h_lines(cls, line_1: Line, line_2: Line, minimal: bool = False) -> "Cell":
        """
        Generate cell from two lines
        :param line_1: first line
        :param line_2: second line
        :param minimal: boolean indicating if cell should be as the larger intersection of both lines
        :return: Cell object
        """
        if minimal:
            _x1 = max(line_1.x1, line_2.x1)
            _x2 = min(line_1.x2, line_2.x2)
        else:
            _x1 = min(line_1.x1, line_2.x1)
            _x2 = max(line_1.x2, line_2.x2)
        _y1 = min(line_1.y1, line_2.y1)
        _y2 = max(line_1.y2, line_2.y2)

        return cls(x1=_x1, x2=_x2, y1=_y1, y2=_y2)


class Row(TableObject):
    def __init__(self, cells: Union[Cell, List[Cell]]):
        if cells is None:
            raise ValueError("cells parameter is null")
        elif isinstance(cells, Cell):
            self._items = [cells]
        else:
            self._items = cells
        self._contours = []

    @property
    def items(self) -> List[Cell]:
        return self._items

    @property
    def contours(self) -> List[Cell]:
        return self._contours

    @property
    def nb_columns(self) -> int:
        return len(self.items)

    @property
    def x1(self) -> int:
        return min([item.x1 for item in self.items])

    @property
    def x2(self) -> int:
        return max([item.x2 for item in self.items])

    @property
    def y1(self) -> int:
        return min([item.y1 for item in self.items])

    @property
    def y2(self) -> int:
        return max([item.y2 for item in self.items])

    @property
    def v_consistent(self) -> bool:
        return len([cell for cell in self.items if cell.y1 != self.y1 or cell.y2 != self.y2]) == 0

    def add_cells(self, cells: Union[Cell, List[Cell]]) -> "Row":
        """
        Add cells to existing row items
        :param cells: Cell object or list
        :return:
        """
        if isinstance(cells, Cell):
            self._items += [cells]
        else:
            self._items += cells

        return self

    def split_in_rows(self, vertical_delimiters: List[int]) -> List["Row"]:
        """
        Split Row object into multiple objects based on vertical delimiters values
        :param vertical_delimiters: list of vertical delimiters values
        :return: list of splitted Row objects according to delimiters
        """
        # Create list of tuples for vertical boundaries
        row_delimiters = [self.y1] + vertical_delimiters + [self.y2]
        row_boundaries = [(i, j) for i, j in zip(row_delimiters, row_delimiters[1:])]

        # Create new list of rows
        l_new_rows = list()
        for boundary in row_boundaries:
            cells = list()
            for cell in copy.deepcopy(self.items):
                _cell = copy.deepcopy(cell)
                _cell.y1, _cell.y2 = boundary
                cells.append(_cell)
            l_new_rows.append(Row(cells=cells))

        return l_new_rows


class Table(TableObject):
    def __init__(self, rows: Union[Row, List[Row]]):
        if rows is None:
            self._items = []
        elif isinstance(rows, Row):
            self._items = [rows]
        else:
            self._items = rows
        self._title = None
        self._data = None

    @property
    def items(self) -> List[Row]:
        return self._items

    @property
    def title(self) -> str:
        return self._title

    def set_title(self, title: str):
        self._title = title

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def nb_rows(self) -> int:
        if self.height > 0:
            return len(self.items)
        return 0

    @property
    def nb_columns(self) -> int:
        if self.items:
            return self.items[0].nb_columns
        return 0

    @property
    def x1(self) -> int:
        return min([item.x1 for item in self.items])

    @property
    def x2(self) -> int:
        return max([item.x2 for item in self.items])

    @property
    def y1(self) -> int:
        return min([item.y1 for item in self.items])

    @property
    def y2(self) -> int:
        return max([item.y2 for item in self.items])

    def process_data(self):
        """
        Process dataframe from OCR
        :return:
        """
        if self.data is None:
            return None

        # Remove empty rows and columns
        self._data = remove_empty_rows(self._data)
        self._data = remove_empty_columns(self._data)

        # If the dataframe is empty or contains a single cell, set it to None
        if len(self._data) == 0 or (len(self._data) == 1 and self._data.shape[1] == 1):
            self._data = None

    def get_text_ocr(self, ocr_page: OCRPage) -> "Table":
        """
        Retrieve text from OCRPage object and set data attribute with dataframe corresponding to table
        :param ocr_page: OCRPage object
        :return: Table object with data attribute containing dataframe
        """
        # Create dataframe with text values and assign to _data attribute
        df_pd = ocr_page.get_text_table(table=self)
        self._data = df_pd

        # Process dataframe
        self.process_data()

        return self
