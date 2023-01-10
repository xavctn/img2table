# coding: utf-8
import copy
from typing import Union, List

from img2table.tables.objects import TableObject
from img2table.tables.objects.cell import Cell


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
    def v_consistent(self) -> bool:
        """
        Indicate if the row is vertically consistent (i.e all cells in row have the same vertical position)
        :return: boolean indicating if the row is vertically consistent
        """
        return all(map(lambda x: (x.y1 == self.y1) and (x.y2 == self.y2), self.items))

    def add_cells(self, cells: Union[Cell, List[Cell]]) -> "Row":
        """
        Add cells to existing row items
        :param cells: Cell object or list
        :return: Row object with cells added
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
            for cell in self.items:
                _cell = copy.deepcopy(cell)
                _cell.y1, _cell.y2 = boundary
                cells.append(_cell)
            l_new_rows.append(Row(cells=cells))

        return l_new_rows

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            try:
                assert self.items == other.items
                return True
            except AssertionError:
                return False
        return False
