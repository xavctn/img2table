# coding: utf-8
import copy
import math
import statistics
from collections import OrderedDict
from typing import Union, List

import numpy as np
import pandas as pd

from img2table.objects.ocr import OCRPage
from img2table.utils.data_processing import split_lines, remove_empty_rows, remove_empty_columns
from img2table.utils.header import detect_header


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
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1


class Line(TableObject):
    def __init__(self, line: Union[np.ndarray, tuple] = None, x1: int = None, y1: int = None, x2: int = None,
                 y2: int = None):
        if line is not None:
            self.x1 = line[0]
            self.y1 = line[1]
            self.x2 = line[2]
            self.y2 = line[3]
        else:
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
            self.y2 = self.y1
        elif abs(self.angle - 90) <= 5:
            self.x2 = self.x1


class Cell(TableObject):
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2

    @classmethod
    def from_lines(cls, line_1: Line, line_2: Line):
        """
        Generate cell from two lines
        :param line_1: first line
        :param line_2: second line
        :return: Cell object
        """
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

    def add_contours(self, contours: Union[Cell, List[Cell]], replace: bool = False):
        """
        Add contours to Row object
        :param contours: contours as Cell objects
        :param replace: boolean indicating to replace existing contours
        :return:
        """
        if replace:
            self._contours = []

        if isinstance(contours, Cell):
            self._contours += [contours]
        else:
            self._contours += contours

    def normalize(self, x1: int = None, x2: int = None):
        """
        Normalize left and right bounds of each cell in the row
        :param x1: left bound
        :param x2: right bound
        :return: Row object
        """
        _x1 = x1 or self.x1
        _x2 = x2 or self.x2
        # Normalize left and right borders of cells
        _cells = list()
        for cell in self.items:
            _cell = Cell(x1=_x1, x2=_x2, y1=cell.y1, y2=cell.y2)
            _cells.append(_cell)
        self._items = _cells
        return self

    @classmethod
    def from_horizontal_lines(cls, line_1: Line, line_2: Line):
        """
        Generate row from horizontal lines
        :param line_1: first horizontal line
        :param line_2: second horizontal line
        :return: Row object
        """
        # Create new cell and instantiate new row
        cell = Cell.from_lines(line_1=line_1, line_2=line_2)
        return cls(cells=cell)

    def split_in_columns(self, column_delimiters: List[int]):
        """
        Split row cell into multiple columns using column delimiters values
        :param column_delimiters: list of column delimiters values
        :return: Row object with splitted columns
        """
        if len(column_delimiters) == 0:
            return self

        # Check if column delimiters are relevant
        if not min([self.x1 <= delimiter <= self.x2 for delimiter in column_delimiters]):
            raise ValueError("Column delimiters are outside of the row bounding box")

        # Check if columns already exists (i.e multiple cells in row)
        if self.nb_columns > 1:
            raise ValueError("Already existing columns in row")

        # Sort column delimiters
        column_delimiters = sorted(column_delimiters)

        # Create list of tuples for column boundaries
        col_delimiters = [self.x1] + column_delimiters + [self.x2]
        col_boundaries = [(i, j) for i, j in zip(col_delimiters, col_delimiters[1:])]

        # Create new cells splitted with boundaries
        new_cells = list()
        for boundary in col_boundaries:
            cell = Cell(x1=boundary[0], x2=boundary[1], y1=self.y1, y2=self.y2)
            new_cells.append(cell)

        self._items = new_cells

        return self

    def split_in_rows(self, vertical_delimiters: List[int]):
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

    def add_row(self, rows: Union[Row, List[Row]]):
        """
        Add row to existing table items
        :param rows: Row object or list
        :return:
        """
        if isinstance(rows, Row):
            self._items += [rows]
        else:
            self._items += rows

        return self

    @classmethod
    def from_horizontal_lines(cls, line_1: Line, line_2: Line):
        """
        Generate table from horizontal lines
        :param line_1: first horizontal line
        :param line_2: second horizontal line
        :return: Table object with one Row object between horizontal lines
        """
        # Create new cell and instantiate new row
        row = Row.from_horizontal_lines(line_1=line_1, line_2=line_2)
        return cls(rows=row)

    def normalize(self):
        """
        Normalize boundaries of rows in table
        :return: Table object with normalized rows
        """
        # Remove empty rows
        self._items = [row for row in self.items if row.height > 0]

        # Normalize left and right borders of rows
        _rows = [row.normalize(x1=self.x1, x2=self.x2) for row in self.items]
        self._items = _rows

        return self

    def split_in_columns(self, column_delimiters: List[int]):
        """
        Split table rows into multiple columns using column delimiters values
        :param column_delimiters: list of column delimiters values
        :return: Table object with rows splitted into columns
        """
        if len(column_delimiters) == 0:
            return self

        _rows = [row.split_in_columns(column_delimiters=column_delimiters) for row in self.items]

        self._items = _rows

        return self

    def process_data(self, with_header: bool = False):
        """
        Process dataframe from OCR
        :param with_header: indicate if the first row of the dataframe is a header
        :return:
        """
        if self.data is None:
            return None

        # Split lines if needed
        self._data = split_lines(self._data)

        # Remove empty rows and columns
        self._data = remove_empty_rows(self._data)
        self._data = remove_empty_columns(self._data)

        # If the dataframe is empty or contains a single cell, set it to None
        if len(self._data) == 0 or (len(self._data) == 1 and self._data.shape[1] == 1):
            self._data = None

        if with_header and self._data is not None:
            # Create new dataframe with first row as header
            first_row = self._data.iloc[0].values
            cols = list(OrderedDict.fromkeys(first_row))

            new_table = self._data.iloc[1:, :len(cols)]
            new_table.columns = cols

            self._data = new_table

    def get_text_ocr(self, ocr_page: OCRPage, img: np.ndarray, header_detection: bool = True):
        """
        Retrieve text from OCRPage object and set data attribute with dataframe corresponding to table
        :param ocr_page: OCRPage object
        :param img: image array
        :param header_detection: boolean indicating if header detection is performed
        :return: Table object with data attribute containing dataframe
        """
        with_header = False
        if header_detection:
            # Detect if table has header
            with_header = detect_header(img=img, ocr_page=ocr_page, table=self)

        # Parse OCR page for each cell of each row
        text_values = [[ocr_page.get_text_cell(cell) for cell in row.items]
                       for row in self.items]

        # Create dataframe from values and assign to _data attribute
        df_pd = pd.DataFrame(text_values)
        self._data = df_pd

        # Process dataframe
        self.process_data(with_header=with_header)

        return self

    def get_text_size(self, ocr_page: OCRPage) -> float:
        """
        Get average text size in the table
        :param ocr_page: OCRPage object
        :return: average text size in the table
        """
        # Get list of text sizes
        text_sizes = ocr_page.get_text_sizes(cell=self)

        # Compute average text size
        return statistics.mean(text_sizes) if text_sizes else None
