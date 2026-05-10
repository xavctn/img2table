from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol

import numpy as np

from img2table.tables.extraction import BBox, ExtractedTable, TableCell

if TYPE_CHECKING:
    from img2table.ocr._types import OCRData


class CoordinateProvider(Protocol):
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class TableObject:
    @cached_property
    def height(self: CoordinateProvider) -> int:
        return self.y2 - self.y1

    @cached_property
    def width(self: CoordinateProvider) -> int:
        return self.x2 - self.x1

    @cached_property
    def area(self) -> int:
        return self.height * self.width


@dataclass
class Line(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: int = 1

    @property
    def angle(self) -> float:
        delta_x = self.x2 - self.x1
        delta_y = self.y2 - self.y1

        return math.atan2(delta_y, delta_x) * 180 / np.pi

    @property
    def length(self) -> float:
        return np.sqrt(self.height**2 + self.width**2)

    @cached_property
    def horizontal(self) -> bool:
        return self.angle % 180 == 0

    @cached_property
    def vertical(self) -> bool:
        return self.angle % 180 == 90

    def reprocess(self) -> Line:
        # Reallocate coordinates in proper order
        _x1 = min(self.x1, self.x2)
        _x2 = max(self.x1, self.x2)
        _y1 = min(self.y1, self.y2)
        _y2 = max(self.y1, self.y2)
        self.x1, self.x2, self.y1, self.y2 = _x1, _x2, _y1, _y2

        # Correct "almost" horizontal or vertical rows
        if abs(self.angle) <= 5:
            y_val = round((self.y1 + self.y2) / 2)
            self.y2 = self.y1 = y_val
        elif abs(self.angle - 90) <= 5:
            x_val = round((self.x1 + self.x2) / 2)
            self.x2 = self.x1 = x_val

        return self

    def __hash__(self) -> int:
        return hash((self.x1, self.y1, self.x2, self.y2))


@dataclass
class Cell(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int
    content: str | None = None

    def bbox(self, margin: int = 0) -> tuple[int, int, int, int]:
        """
        Return bounding box corresponding to the object
        :param margin: general margin used for the bounding box
        :return: tuple representing a bounding box
        """
        return (self.x1 - margin, self.y1 - margin, self.x2 + margin, self.y2 + margin)

    def table_cell(self, image_width: int, image_height: int) -> TableCell:
        bbox = BBox(
            x1=self.x1,
            x2=self.x2,
            y1=self.y1,
            y2=self.y2,
            image_width=image_width,
            image_height=image_height,
        )
        return TableCell(bbox=bbox, value=self.content)

    def __hash__(self) -> int:
        return hash((self.x1, self.y1, self.x2, self.y2))


@dataclass
class Row(TableObject):
    cells: list[Cell] = field(default_factory=list)

    @property
    def nb_columns(self) -> int:
        return len(self.cells)

    @property
    def x1(self) -> int:
        return min((x.x1 for x in self.cells), default=0)

    @property
    def x2(self) -> int:
        return max((x.x2 for x in self.cells), default=0)

    @property
    def y1(self) -> int:
        return min((x.y1 for x in self.cells), default=0)

    @property
    def y2(self) -> int:
        return max((x.y2 for x in self.cells), default=0)

    @property
    def v_consistent(self) -> bool:
        """
        Indicate if the row is vertically consistent (i.e all cells in row have the same vertical position)
        :return: boolean indicating if the row is vertically consistent
        """
        return all((x.y1 == self.y1) and (x.y2 == self.y2) for x in self.cells)

    def add_cells(self, cells: list[Cell]) -> Row:
        """
        Add cells to existing row items
        :param cells: Cell list
        :return: Row object with cells added
        """
        self.cells += cells
        return self

    def __hash__(self) -> int:
        return hash(tuple(self.cells))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Row) and self.cells == other.cells


@dataclass
class Table(TableObject):
    rows: list[Row] = field(default_factory=list)
    title_area: Cell | None = None
    title: str | None = None

    def set_title_area(self, title_area: Cell) -> None:
        self.title_area = title_area

    def set_title(self, title: str | None) -> None:
        self.title = title

    @property
    def nb_rows(self) -> int:
        return len(self.rows)

    @property
    def nb_columns(self) -> int:
        return self.rows[0].nb_columns if self.rows else 0

    @property
    def nb_cells(self) -> int:
        return len({cell for row in self.rows for cell in row.cells})

    @property
    def x1(self) -> int:
        return min((x.x1 for x in self.rows), default=0)

    @property
    def x2(self) -> int:
        return max((x.x2 for x in self.rows), default=0)

    @property
    def y1(self) -> int:
        return min((x.y1 for x in self.rows), default=0)

    @property
    def y2(self) -> int:
        return max((x.y2 for x in self.rows), default=0)

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    @cached_property
    def lines(self) -> list[Line]:
        # Create lines from cells
        h_lines, v_lines = [], []
        for cell in [cell for row in self.rows for cell in row.cells]:
            # Add vertical lines
            v_lines += [
                Line(x1=cell.x1, y1=cell.y1, x2=cell.x1, y2=cell.y2),
                Line(x1=cell.x2, y1=cell.y1, x2=cell.x2, y2=cell.y2),
            ]
            # Add horizontal lines
            h_lines += [
                Line(x1=cell.x1, y1=cell.y1, x2=cell.x2, y2=cell.y1),
                Line(x1=cell.x1, y1=cell.y2, x2=cell.x2, y2=cell.y2),
            ]

        # Merge vertical lines
        seq = iter(sorted(v_lines, key=lambda ln: (ln.x1, ln.y1)))
        v_lines_groups = [[next(seq)]]
        for line in seq:
            prev_line = v_lines_groups[-1][-1]
            if line.x1 > prev_line.x1 or line.y1 > prev_line.y2:
                v_lines_groups.append([])
            v_lines_groups[-1].append(line)

        # Merge horizontal lines
        seq = iter(sorted(h_lines, key=lambda ln: (ln.y1, ln.x1)))
        h_lines_groups = [[next(seq)]]
        for line in seq:
            prev_line = h_lines_groups[-1][-1]
            if line.y1 > prev_line.y1 or line.x1 > prev_line.x2:
                h_lines_groups.append([])
            h_lines_groups[-1].append(line)

        return [
            Line(
                x1=min(ln.x1 for ln in gp),
                y1=min(ln.y1 for ln in gp),
                x2=max(ln.x2 for ln in gp),
                y2=max(ln.y2 for ln in gp),
            )
            for gp in v_lines_groups + h_lines_groups
        ]

    def remove_rows(self, row_ids: list[int]) -> None:
        """
        Remove rows by ids
        :param row_ids: list of row ids to be removed
        """
        # Get remaining rows
        remaining_rows = [idx for idx in range(self.nb_rows) if idx not in row_ids]

        if len(remaining_rows) > 1:
            # Check created gaps between rows
            gaps = [
                (id_row, id_next)
                for id_row, id_next in pairwise(remaining_rows)
                if id_next - id_row > 1
            ]

            for id_row, id_next in gaps:
                # Normalize y value between rows
                y_gap = round((self.rows[id_row].y2 + self.rows[id_next].y1) / 2)

                # Put y value in both rows
                for c in self.rows[id_row].cells:
                    c.y2 = max(c.y2, y_gap)
                for c in self.rows[id_next].cells:
                    c.y1 = min(c.y1, y_gap)

        # Remove rows
        for idx in reversed(row_ids):
            self.rows.pop(idx)

    def remove_columns(self, col_ids: list[int]) -> None:
        """
        Remove columns by ids
        :param col_ids: list of column ids to be removed
        """
        # Get remaining cols
        remaining_cols = [idx for idx in range(self.nb_columns) if idx not in col_ids]

        if len(remaining_cols) > 1:
            # Check created gaps between columns
            gaps = [
                (id_col, id_next)
                for id_col, id_next in pairwise(remaining_cols)
                if id_next - id_col > 1
            ]

            for id_col, id_next in gaps:
                # Normalize x value between columns
                x_gap = round(
                    np.mean([row.cells[id_col].x2 + row.cells[id_next].x1 for row in self.rows]) / 2
                )

                # Put x value in both columns
                for row in self.rows:
                    row.cells[id_col].x2 = max(row.cells[id_col].x2, x_gap)
                    row.cells[id_next].x1 = min(row.cells[id_next].x1, x_gap)

        # Remove columns
        for idx in reversed(col_ids):
            for id_row in range(self.nb_rows):
                self.rows[id_row].cells.pop(idx)

    def get_content(
        self, ocr_data: OCRData, page_number: int | None = None, min_confidence: int = 50
    ) -> Table:
        """
        Retrieve text from OCRData object and reprocess table to remove empty rows / columns
        :param ocr_data: OCRData object
        :param page_number: page number in OCRData object
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: Table object with data attribute containing dataframe
        """
        # Get content for each cell
        self = ocr_data.get_text_table(  # noqa: PLW0642
            table=self, page_number=page_number, min_confidence=min_confidence
        )

        # Check for empty rows and remove if necessary
        empty_rows = []
        for idx, row in enumerate(self.rows):
            if all(c.content is None for c in row.cells):
                empty_rows.append(idx)
        self.remove_rows(row_ids=empty_rows)

        # Check for empty columns and remove if necessary
        empty_cols = []
        for idx in range(self.nb_columns):
            col_cells = [row.cells[idx] for row in self.rows]
            if all(c.content is None for c in col_cells):
                empty_cols.append(idx)
        self.remove_columns(col_ids=empty_cols)

        # Check for uniqueness of content
        unique_cells = {cell for row in self.rows for cell in row.cells}
        if len(unique_cells) == 1:
            self.rows = [Row(cells=[self.rows[0].cells[0]])]

        return self

    def overlaps(self, other: Table, pct: float = 0.5) -> bool:
        """
        Check if this table overlaps with another table.
        :param other: The other table to check for overlap.
        :param pct: The minimum percentage of overlap required.
        :return: True if the tables overlap, False otherwise.
        """
        # Compute intersection area
        intersection_area = max(0, min(self.x2, other.x2) - max(self.x1, other.x1)) * max(
            0, min(self.y2, other.y2) - max(self.y1, other.y1)
        )

        return intersection_area / min(self.area, other.area) >= pct

    def has_valid_shape(self) -> bool:
        if min(self.nb_rows, self.nb_columns) < 2 or self.nb_cells < 4:
            return False

        # Check coherency of structure
        cells = sorted(
            {cell for row in self.rows for cell in row.cells},
            key=lambda cell: cell.area,
            reverse=True,
        )
        if len({c.x1 for c in cells} | {c.x2 for c in cells}) > self.nb_columns + 1:
            return False
        if len({c.y1 for c in cells} | {c.y2 for c in cells}) > self.nb_rows + 1:
            return False

        # Check overlap between cells
        cells_array = np.array([[cell.x1, cell.y1, cell.x2, cell.y2] for cell in cells])
        x_overlap = np.maximum(
            0,
            np.minimum(cells_array[:, 2], cells_array[:, 2][:, None])
            - np.maximum(cells_array[:, 0], cells_array[:, 0][:, None]),
        )
        y_overlap = np.maximum(
            0,
            np.minimum(cells_array[:, 3], cells_array[:, 3][:, None])
            - np.maximum(cells_array[:, 1], cells_array[:, 1][:, None]),
        )
        overlap_area = np.sum(np.multiply(x_overlap, y_overlap)) - sum(cell.area for cell in cells)

        return overlap_area < 0.2 * self.area

    def extracted_table(self, image_width: int, image_height: int) -> ExtractedTable:
        bbox = BBox(
            x1=self.x1,
            x2=self.x2,
            y1=self.y1,
            y2=self.y2,
            image_width=image_width,
            image_height=image_height,
        )
        content = OrderedDict(
            {
                idx: [
                    cell.table_cell(image_width=image_width, image_height=image_height)
                    for cell in row.cells
                ]
                for idx, row in enumerate(self.rows)
            }
        )
        return ExtractedTable(bbox=bbox, title=self.title, content=content)

    def __hash__(self) -> int:
        return hash(tuple(self.rows))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Table) and self.rows == other.rows and self.title == other.title
