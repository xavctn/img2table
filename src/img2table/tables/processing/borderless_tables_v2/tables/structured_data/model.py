from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise

import numpy as np

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
from img2table.tables.processing.common import _cluster_values


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

    def _evaluate_key_separation_value(
        self, rows: list[MergedRow], ref_separation: float
    ) -> tuple[float, list]:
        """
        Evaluate pertinence of separation value based on created rows consistency
        :param rows: list of merged rows from the section
        :param ref_separation: reference separation value
        :return: key consistency score and created ranges
        """
        # Create sections
        sections = [[rows[0]]]
        for row in rows[1:]:
            prev_row = sections[-1][-1]
            separation = row.y_center - prev_row.y_center
            if separation <= 0.75 * ref_separation:
                sections[-1].append(row)
            else:
                sections.append([row])

        # Define ranges
        ranges = []
        current_y = self.y_min
        for prev_section, nxt_section in pairwise(sections):
            # Get middle point to compute separator between sections
            middle_point = int((prev_section[-1].y2 + nxt_section[0].y1) / 2)
            ranges.append((current_y, middle_point))
            current_y = middle_point

        # Add last range
        ranges.append((current_y, self.y_max))

        # Compute range heights
        range_heights = [rng[1] - rng[0] for rng in ranges]

        return 1 - np.std(range_heights) / self.height if len(ranges) > 1 else 0, ranges

    def row_ranges(self) -> list[tuple[int, int]]:
        """
        Identify vertical position ranges corresponding to rows in a section
        :return: list of (start, end) vertical ranges
        """
        if self._row_ranges is None:
            # Compute bounds and merged rows
            rows = self.merged_rows

            if len(rows) <= 1:
                self._row_ranges = [(self.y_min, self.y_max)]
                return self._row_ranges

            # Compute separation between elements
            separations: list[float] = [nxt.y_center - prv.y_center for prv, nxt in pairwise(rows)]
            median_sep = np.median(separations)
            median_row_height = np.median([row.height for row in rows])

            # Cluster separations to identify distinct spacing patterns
            cluster_labels = _cluster_values(values=separations, median_gap_multiple=3)

            # Find the most common cluster that corresponds to rows to get eligible separations
            eligible_separations = {median_sep}
            for cluster_id, _ in Counter(cluster_labels).most_common(2):
                cluster_separations = [
                    sep
                    for sep, label in zip(separations, cluster_labels, strict=True)
                    if label == cluster_id
                ]
                if (cluster_median_sep := np.median(cluster_separations)) > median_row_height:
                    eligible_separations.add(cluster_median_sep)

            # Evaluate best separation value
            best_score, best_ranges = 0, []
            for ref_sep in eligible_separations:
                score, ranges = self._evaluate_key_separation_value(
                    rows=rows, ref_separation=ref_sep
                )
                if score > best_score:
                    best_score = score
                    best_ranges = ranges

            self._row_ranges = best_ranges or [(self.y_min, self.y_max)]
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
