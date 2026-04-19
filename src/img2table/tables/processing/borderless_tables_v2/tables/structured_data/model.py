from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables_v2._model import (
    ColumnSection,
    MergedRow,
    Whitespace,
    identify_merged_rows,
)
from img2table.tables.processing.borderless_tables_v2.tables.structured_data.metrics import (
    TableMetrics,
)
from img2table.tables.processing.common import compute_row_ranges


@dataclass
class StructuredSection:
    height: int
    width: int
    char_length: float
    items: list[Cell]
    whitespaces: list[Whitespace]
    _row_ranges: list[tuple[int, int]] | None = None

    @classmethod
    def from_section(
        cls, section: ColumnSection, width: int, height: int, char_length: float
    ) -> "StructuredSection":
        """
        Create a StructuredSection from a ColumnSection.
        :param section: the column section to convert
        :param width: full page width in pixels
        :param height: full page height in pixels
        :param char_length: Character length in pixels.
        :return: Structured section with normalized coordinates
        """
        return cls(
            height=height,
            width=width,
            char_length=char_length,
            items=section.items,
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

    @cached_property
    def merged_rows(self) -> list[MergedRow]:
        return identify_merged_rows(cnts=self.items)

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
            self._row_ranges = compute_row_ranges(rows=self.merged_rows, y_min=self.y_min, y_max=self.y_max)
        return self._row_ranges

    def table_score(self) -> float:
        """
        Compute weighted table confidence score.
        :return: score between 0 and 1
        """
        # Compute metrics
        metrics = TableMetrics.from_section(section=self)

        # Hard reject rules
        if len(self.merged_rows) < 3:
            # Not enough base rows (before row merging)
            return 0.0
        if self.nb_columns < 2 or self.nb_rows < 2 or max(self.nb_rows, self.nb_columns) < 3:
            # Not enough columns / Not enough rows / At least 3 rows or columns required
            return 0.0
        if sum(ratio >= 0.5 for ratio in metrics.presence_ratios) < 2:
            # Insufficient column presence
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
        return self.table_score() >= 0.425

    def table(self) -> Table:
        """
        Create Table object from section
        :return: Table object
        """
        # Compute x delimiters
        x_delimiters = [
            self.whitespaces[0].end,
            *[(ws.start + ws.end) // 2 for ws in self.whitespaces[1:-1]],
            self.whitespaces[-1].start,
        ]

        # Create rows
        rows = [
            Row(
                cells=[
                    Cell(x1=x_start, y1=y_start, x2=x_end, y2=y_end)
                    for x_start, x_end in pairwise(x_delimiters)
                ]
            )
            for y_start, y_end in self.row_ranges()
        ]

        return Table(rows=rows)
