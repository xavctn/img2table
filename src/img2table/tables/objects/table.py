from __future__ import annotations

import typing
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise

import numpy as np

from img2table.tables.objects import TableObject
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.extraction import BBox, ExtractedTable
from img2table.tables.objects.line import Line
from img2table.tables.objects.row import Row

if typing.TYPE_CHECKING:
    from img2table.ocr.data import OCRDataframe


@dataclass
class Table(TableObject):
    def __init__(self, rows: Row | list[Row], borderless: bool = False) -> None:
        if rows is None:
            self.items = []
        elif isinstance(rows, Row):
            self.items = [rows]
        else:
            self.items = rows
        self.title: str | None = None
        self.borderless = borderless

    def set_title(self, title: str | None) -> None:
        self.title = title

    @property
    def nb_rows(self) -> int:
        return len(self.items)

    @property
    def nb_columns(self) -> int:
        return self.items[0].nb_columns if self.items else 0

    @property
    def x1(self) -> int:
        return min((x.x1 for x in self.items), default=0)

    @property
    def x2(self) -> int:
        return max((x.x2 for x in self.items), default=0)

    @property
    def y1(self) -> int:
        return min((x.y1 for x in self.items), default=0)

    @property
    def y2(self) -> int:
        return max((x.y2 for x in self.items), default=0)

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    @cached_property
    def lines(self) -> list[Line]:
        # Create lines from cells
        h_lines, v_lines = [], []
        for cell in [cell for row in self.items for cell in row.items]:
            # Add vertical lines
            v_lines.append(Line(x1=cell.x1, y1=cell.y1, x2=cell.x1, y2=cell.y2))
            v_lines.append(Line(x1=cell.x2, y1=cell.y1, x2=cell.x2, y2=cell.y2))
            # Add horizontal lines
            h_lines.append(Line(x1=cell.x1, y1=cell.y1, x2=cell.x2, y2=cell.y1))
            h_lines.append(Line(x1=cell.x1, y1=cell.y2, x2=cell.x2, y2=cell.y2))

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
                x1=min([ln.x1 for ln in gp]),
                y1=min([ln.y1 for ln in gp]),
                x2=max([ln.x2 for ln in gp]),
                y2=max([ln.y2 for ln in gp]),
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
                y_gap = round((self.items[id_row].y2 + self.items[id_next].y1) / 2)

                # Put y value in both rows
                for c in self.items[id_row].items:
                    c.y2 = max(c.y2, y_gap)
                for c in self.items[id_next].items:
                    c.y1 = min(c.y1, y_gap)

        # Remove rows
        for idx in reversed(row_ids):
            self.items.pop(idx)

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
                    np.mean([row.items[id_col].x2 + row.items[id_next].x1 for row in self.items])
                    / 2
                )

                # Put x value in both columns
                for row in self.items:
                    row.items[id_col].x2 = max(row.items[id_col].x2, x_gap)
                    row.items[id_next].x1 = min(row.items[id_next].x1, x_gap)

        # Remove columns
        for idx in reversed(col_ids):
            for id_row in range(self.nb_rows):
                self.items[id_row].items.pop(idx)

    def get_content(self, ocr_df: OCRDataframe, min_confidence: int = 50) -> Table:
        """
        Retrieve text from OCRDataframe object and reprocess table to remove empty rows / columns
        :param ocr_df: OCRDataframe object
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: Table object with data attribute containing dataframe
        """
        # Get content for each cell
        self = ocr_df.get_text_table(table=self, min_confidence=min_confidence)  # noqa: PLW0642

        # Check for empty rows and remove if necessary
        empty_rows = []
        for idx, row in enumerate(self.items):
            if all(c.content is None for c in row.items):
                empty_rows.append(idx)
        self.remove_rows(row_ids=empty_rows)

        # Check for empty columns and remove if necessary
        empty_cols = []
        for idx in range(self.nb_columns):
            col_cells = [row.items[idx] for row in self.items]
            if all(c.content is None for c in col_cells):
                empty_cols.append(idx)
        self.remove_columns(col_ids=empty_cols)

        # Check for uniqueness of content
        unique_cells = {cell for row in self.items for cell in row.items}
        if len(unique_cells) == 1:
            self._items = [Row(cells=self.items[0].items[0])]

        return self

    @property
    def extracted_table(self) -> ExtractedTable:
        bbox = BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
        content = OrderedDict(
            {idx: [cell.table_cell for cell in row.items] for idx, row in enumerate(self.items)}
        )
        return ExtractedTable(bbox=bbox, title=self.title, content=content)

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

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: object) -> bool:
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
