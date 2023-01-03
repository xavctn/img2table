# coding: utf-8
import math
from typing import Union, List

import pandas as pd

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects import TableObject
from img2table.tables.objects.row import Row
from img2table.utils.data_processing import remove_empty_rows, remove_empty_columns


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
        return len(self.items) if self.height > 0 else 0

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
        if math.prod(self.data.shape) <= 1:
            self._data = None

    def get_content(self, ocr_df: OCRDataframe) -> "Table":
        """
        Retrieve text from OCRDataframe object and set data attribute with dataframe corresponding to table
        :param ocr_df: OCRDataframe object
        :return: Table object with data attribute containing dataframe
        """
        # Create dataframe with text values and assign to _data attribute
        df_pd = ocr_df.get_text_table(table=self)
        self._data = df_pd

        # Process dataframe
        self.process_data()

        return self
