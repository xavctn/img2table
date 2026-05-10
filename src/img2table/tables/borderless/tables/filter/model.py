from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise

import numpy as np

from img2table.tables.borderless.tables.filter.metrics import (
    TableMetrics,
)
from img2table.tables.borderless.types import (
    ColumnSection,
    MergedRow,
    Whitespace,
)
from img2table.tables.common import compute_row_ranges
from img2table.tables.types import Cell, Row, Table


@dataclass
class StructuredSection:
    height: int = field(repr=False)
    width: int = field(repr=False)
    char_length: float
    items: list[Cell] = field(repr=False)
    merged_rows: list[MergedRow] = field(repr=False)
    whitespaces: list[Whitespace]
    _row_ranges: list[tuple[int, int]] | None = None
    idx: int = field(default=0, repr=False)

    @classmethod
    def from_section(
        cls, idx: int, section: ColumnSection, width: int, height: int, char_length: float
    ) -> StructuredSection:
        """
        Create a StructuredSection from a ColumnSection.
        :param idx: index of the column section
        :param section: the column section to convert
        :param width: full page width in pixels
        :param height: full page height in pixels
        :param char_length: Character length in pixels.
        :return: Structured section with normalized coordinates
        """
        return cls(
            idx=idx,
            height=height,
            width=width,
            char_length=char_length,
            items=section.items,
            merged_rows=section.rows,
            whitespaces=section.whitespaces,
        )

    @cached_property
    def x_min(self) -> int:
        return min((item.x1 for item in self.items), default=0)

    @cached_property
    def x_max(self) -> int:
        return max((item.x2 for item in self.items), default=0)

    @cached_property
    def y_min(self) -> int:
        return min((item.y1 for item in self.items), default=0)

    @cached_property
    def y_max(self) -> int:
        return max((item.y2 for item in self.items), default=0)

    @property
    def nb_columns(self) -> int:
        return len(self.whitespaces) - 1

    @property
    def nb_rows(self) -> int:
        return len(self.row_ranges())

    @property
    def cols(self) -> list[tuple[int, ...]]:
        return [(prv.end, nxt.start) for prv, nxt in pairwise(self.whitespaces)]

    @cached_property
    def col_cells(self) -> list[list[Cell]]:
        """
        Extract per-column cell lists from a section by assigning each cell to its column based on x-center
        :return: list of cell lists, one per column
        """
        col_rows = []
        for start, end in self.cols:
            # Identify column items
            col_items = sorted(
                [item for item in self.items if item.x1 >= start and item.x2 <= end],
                key=lambda cell: cell.y1 + cell.y2,
            )
            col_rows.append(col_items)

        return col_rows

    def row_ranges(self) -> list[tuple[int, int]]:
        """
        Identify vertical position ranges corresponding to rows in a section
        :return: list of (start, end) vertical ranges
        """
        if self._row_ranges is None:
            # Compute row ranges
            self._row_ranges = compute_row_ranges(
                rows=self.merged_rows, y_min=self.y_min, y_max=self.y_max
            )
        return self._row_ranges

    @property
    def row_height(self) -> float:
        if self.is_structured():
            return np.mean([y_end - y_start for y_start, y_end in self.row_ranges()])
        return 0.0

    @cached_property
    def table_score(self) -> float:  # noqa: PLR0911
        """
        Compute weighted table confidence score.
        :return: score between 0 and 1
        """
        # Hard reject rules
        if len(self.merged_rows) < 3:
            # Not enough base rows (before row merging)
            return 0.0
        if self.nb_columns < 2 or self.nb_rows < 2 or max(self.nb_rows, self.nb_columns) < 3:
            # Not enough columns / Not enough rows / At least 3 rows or columns required
            return 0.0

        # Compute metrics
        metrics = TableMetrics.from_section(section=self)

        if sum(ratio >= 0.5 for ratio in metrics.presence_ratios) < 2:
            # Insufficient column presence
            return 0.0
        if metrics.spacing_consistency < 0.25:
            # Insufficient spacing consistency
            return 0.0
        if (
            metrics.full_text >= 2 / 3
            and np.mean([it.width for it in self.items]) >= 25 * self.char_length
        ):
            # Most likely text columns
            return 0.0
        if metrics.network_connectivity < 0.35:
            # Weak network connectivity
            return 0.0
        if metrics.mean_column_alignment < 0.35 and metrics.spacing_consistency < 0.35:
            # Weak alignment and spacing
            return 0.0

        # Return score
        return metrics.score()

    def is_structured(self) -> bool:
        """
        Assess whether the section behaves like a table
        :return: True if the table is structured, False otherwise
        """
        return self.table_score >= 0.425

    def table(
        self,
        x_min: int | None = None,
        x_max: int | None = None,
        y_min: int | None = None,
        y_max: int | None = None,
        ref_whitespaces: list[Whitespace] | None = None,
    ) -> Table:
        """
        Create Table object from section
        :param x_min: left bound for the table
        :param x_max: right bound for the table
        :param y_min: upper bound for the table
        :param y_max: lower bound for the table
        :param ref_whitespaces: list of reference whitespaces to follow
        :return: Table object
        """
        # Fill reference whitespaces if not provided
        ref_whitespaces = sorted(ref_whitespaces or self.whitespaces, key=lambda ws: ws.start)

        # Compute column ranges from reference separators
        ref_ranges = [(prv.end, nxt.start) for prv, nxt in pairwise(ref_whitespaces)]

        # Keep only own whitespaces that overlap any reference whitespaces
        kept_ws = []
        for ws in self.whitespaces:
            overlapping_ws = [ref_ws for ref_ws in ref_whitespaces if ws.overlaps(ref_ws)]
            if not overlapping_ws:
                continue

            matched_ws = [ws, *overlapping_ws]
            kept_ws.append(
                Whitespace(
                    start=max(item.start for item in matched_ws),
                    end=min(item.end for item in matched_ws),
                    start_bound=all(item.start_bound for item in matched_ws),
                    end_bound=all(item.end_bound for item in matched_ws),
                )
            )

        # Get created horizontal delimiters
        x_delimiters = [
            x_min if x_min is not None else kept_ws[0].end,
            *[(ws.start + ws.end) // 2 for ws in kept_ws[1:-1]],
            x_max if x_max is not None else kept_ws[-1].start,
        ]

        # Compute relevant column ranges (duplicate multiple times the range if it overlaps with multiple reference ranges)
        column_ranges = [
            max(
                pairwise(x_delimiters),
                key=lambda rng: min(rng[1], end) - max(rng[0], start),
            )
            for start, end in ref_ranges
        ]

        # Compute y delimiters
        row_delimiters = sorted({y for rng in self.row_ranges() for y in rng})
        y_delimiters = [
            y_min if y_min is not None else self.y_min,
            *row_delimiters[1:-1],
            y_max if y_max is not None else self.y_max,
        ]

        # Create rows
        rows = [
            Row(
                cells=[
                    Cell(x1=x_start, y1=y_start, x2=x_end, y2=y_end)
                    for x_start, x_end in column_ranges
                ]
            )
            for y_start, y_end in pairwise(y_delimiters)
        ]

        return Table(rows=rows)
