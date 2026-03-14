from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise

import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables_v2._model import (
    ColumnSection,
    MergedRow,
    Whitespace,
    identify_merged_rows,
)


def _median_absolute_difference(values: list[float]) -> float:
    """
    Compute MAD of series
    :param values: list of values
    :return: series MAD
    """
    median = np.median(values)
    abs_diffs = [abs(x - median) for x in values]
    return (np.median(abs_diffs) + np.mean(abs_diffs)) / 2


def _cluster_values(values: list[float]) -> list[int]:
    """
    Cluster values
    :param values: list of float values
    :return: cluster label (0, 1, 2, ...) for each value
    """
    if len(values) <= 1:
        return [0] * len(values)

    # Sort values while tracking original indices
    sorted_with_idx = sorted(enumerate(values), key=lambda x: x[1])
    sorted_values = [val for _, val in sorted_with_idx]

    # Compute gaps between consecutive sorted values
    gaps = [nxt - prv for prv, nxt in pairwise(sorted_values)]
    gap_threshold = 3 * (np.median(gaps) if len(gaps) > 2 else min(gaps))

    # Create clusters
    cluster_id, cluster_labels_sorted = 0, [0]
    for gap in gaps:
        if gap > gap_threshold:
            cluster_id += 1
        cluster_labels_sorted.append(cluster_id)

    # Map back to original order
    cluster_labels = [0] * len(values)
    for i, (orig_idx, _) in enumerate(sorted_with_idx):
        cluster_labels[orig_idx] = cluster_labels_sorted[i]

    return cluster_labels


@dataclass
class StructuredSection:
    char_height: float
    char_width: float
    items: list[Cell]
    whitespaces: list[Whitespace]
    _row_ranges: list[tuple[float, float]] | None = None

    @cached_property
    def x_min(self) -> float:
        return min((item.x1 for item in self.items), default=0)

    @cached_property
    def x_max(self) -> float:
        return max((item.x2 for item in self.items), default=0)

    @cached_property
    def y_min(self) -> float:
        return min((item.y1 for item in self.items), default=0)

    @cached_property
    def y_max(self) -> float:
        return max((item.y2 for item in self.items), default=0)

    @cached_property
    def spacing_regularization(self) -> float:
        return max(0.025, 3 * self.char_height)

    @property
    def nb_columns(self) -> int:
        return len(self.whitespaces) - 1

    @property
    def nb_rows(self) -> int:
        return len(self.row_ranges())

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
        # Normalize items
        norm_items = [
            Cell(
                x1=cell.x1 / width,
                y1=cell.y1 / height,
                x2=cell.x2 / width,
                y2=cell.y2 / height,
                content=cell.content,
            )
            for cell in section.items
        ]
        norm_whitespaces = [
            Whitespace(
                start=ws.start / width,
                end=ws.end / width,
                start_bound=ws.start_bound,
                end_bound=ws.end_bound,
            )
            for ws in section.whitespaces
        ]

        return cls(
            char_height=char_length / height,
            char_width=char_length / width,
            items=norm_items,
            whitespaces=norm_whitespaces,
        )

    @property
    def cols(self) -> list[tuple[float, ...]]:
        return [(prv.end, nxt.start) for prv, nxt in pairwise(self.whitespaces)]

    @cached_property
    def col_cells(self) -> list[list[Cell]]:
        """
        Extract per-column cell lists from a section by assigning each cell to its column based on x-center
        :return: list of cell lists, one per column
        """
        col_rows = []
        for prv, nxt in pairwise(self.whitespaces):
            # Identify column items
            col_items = sorted(
                [item for item in self.items if item.x1 >= prv.end and item.x2 <= nxt.start],
                key=lambda cell: cell.y1 + cell.y2,
            )
            col_rows.append(col_items)

        return col_rows

    @cached_property
    def merged_rows(self) -> list[MergedRow]:
        return identify_merged_rows(cnts=self.items)

    @cached_property
    def row_cells(self) -> list[list[Cell]]:
        rows_by_range = []
        for y_start, y_end in self.row_ranges():
            row_items = [
                cell
                for cell in self.items
                if min(cell.y2, y_end) - max(cell.y1, y_start) >= 0.5 * cell.height
            ]
            rows_by_range.append(sorted(row_items, key=lambda cell: (cell.x1, cell.y1)))
        return rows_by_range

    @cached_property
    def occupancy_matrix(self) -> list[list[bool]]:
        """
        Assign each row item to the column containing its x-center.
        :return: row/column occupancy matrix
        """
        matrix = [[False] * self.nb_columns for _ in self.row_ranges()]
        if self.nb_columns == 0:
            return matrix

        for row_idx, row_items in enumerate(self.row_cells):
            for cell in row_items:
                center = (cell.x1 + cell.x2) / 2
                for col_idx, (start, end) in enumerate(self.cols):
                    if start <= center <= end:
                        matrix[row_idx][col_idx] = True
                        break

        return matrix

    def _col_alignment_score(self, col: list[Cell]) -> float | None:
        """
        Compute alignment score of a specific column
        :param col: list of cells in the column
        :return: alignment score of the column
        """
        # Check column width
        if (
            max((cnt.x2 for cnt in col), default=0) - min((cnt.x1 for cnt in col), default=0)
            < 3 * self.char_width
        ):
            return 0.0

        # Compute merged rows
        rows = identify_merged_rows(cnts=col)

        if len(rows) < 2:
            return None

        alignments = {"left_alignment": [], "center_alignment": [], "right_alignment": []}
        for cell in rows:
            alignments["left_alignment"].append(cell.x1)
            alignments["center_alignment"].append((cell.x1 + cell.x2) / 2)
            alignments["right_alignment"].append(cell.x2)

        # Compute deviations
        deviations = {
            "left_alignment": _median_absolute_difference(alignments["left_alignment"]),
            "center_alignment": _median_absolute_difference(alignments["center_alignment"]),
            "right_alignment": _median_absolute_difference(alignments["right_alignment"]),
        }

        # Get alignment with lowest deviation
        best_alignment = min(deviations, key=lambda k: deviations[k])

        if best_alignment == "center_alignment":
            score = 1 - deviations["center_alignment"] / 0.02
        elif best_alignment == "left_alignment":
            # Check that main alignment is at the left bound
            values, deviation = alignments["left_alignment"], deviations["left_alignment"]
            if (np.median(values) - min(values)) / (max(values) - min(values) + 10e-6) <= 0.05:
                score = 1 - deviation / 0.02
            else:
                # Apply penalization
                score = 1 - (2 * deviation) / 0.02
        else:
            # Check that main alignment is at the right bound
            values, deviation = alignments["right_alignment"], deviations["right_alignment"]
            if (max(values) - np.median(values)) / (max(values) - min(values) + 10e-6) <= 0.05:
                score = 1 - deviation / 0.02
            else:
                # Apply penalization
                score = 1 - 2 * deviation / 0.02

        return max(0.0, score)

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
            middle_point = (prev_section[-1].y2 + nxt_section[0].y1) / 2
            ranges.append((current_y, middle_point))
            current_y = middle_point

        # Add last range
        ranges.append((current_y, self.y_max))

        # Compute range heights
        range_heights = [rng[1] - rng[0] for rng in ranges]

        return 1 - np.std(range_heights) if len(ranges) > 1 else 0, ranges

    def row_ranges(self) -> list[tuple[float, float]]:
        """
        Identify vertical position ranges corresponding to rows in a section
        :return: list of (start, end) vertical ranges
        """
        if self._row_ranges is None:
            # Compute bounds and merged rows
            rows = self.merged_rows

            if len(rows) <= 1:
                return [(self.y_min, self.y_max)]

            # Compute separation between elements
            separations: list[float] = [nxt.y_center - prv.y_center for prv, nxt in pairwise(rows)]
            median_sep = np.median(separations)
            median_row_height = np.median([row.height for row in rows])

            # Cluster separations to identify distinct spacing patterns
            cluster_labels = _cluster_values(separations)

            # Find the most common cluster that corresponds to rows to get eligible separations
            eligible_separations = {median_sep}
            for _, (cluster_id, _) in enumerate(Counter(cluster_labels).most_common(2)):
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

    def content_spacing_consistency(self) -> float:
        """
        Check if vertical spacing between content rows is consistent
        :return: score between 0 and 1
        """
        if not self.items:
            return 0.0

        range_rows, range_heights = [], []
        for y_start, y_end in self.row_ranges():
            range_heights.append(y_end - y_start)
            range_rows.append(
                [
                    row
                    for row in self.merged_rows
                    if min(row.y2, y_end) - max(row.y1, y_start) >= 0.5 * row.height
                ]
            )

        # Compute separations between consecutive rows
        range_seps: list[float] = []
        for prv_rows, nxt_rows in pairwise(range_rows):
            if not prv_rows or not nxt_rows:
                continue
            prv_y = max(row.y_center for row in prv_rows)
            nxt_y = min(row.y_center for row in nxt_rows)
            range_seps.append(nxt_y - prv_y)

        if not range_seps:
            return 0.0

        # Apply penalization
        return max(
            0, 1 - max(np.std(range_seps), np.std(range_heights)) / self.spacing_regularization
        )

    def full_text_score(self) -> float:
        """
        Identify tables where content occupies the entire span of columns
        :return: score between 0 and 1
        """
        nb_rows, nb_full_rows = 0, 0
        for (start, end), cells in zip(self.cols, self.col_cells, strict=True):
            rows = identify_merged_rows(cnts=cells)

            # Update row number
            nb_rows += len(rows)
            nb_full_rows += len([row for row in rows if row.width >= 0.9 * (end - start)])

        return nb_full_rows / nb_rows if nb_rows else 0.0

    def sparsity_score(self) -> float:
        """
        Compute table sparsity score
        :return: score between 0 and 1
        """
        if self.nb_columns * self.nb_rows == 0:
            return 0.0

        nb_used_cells = 0
        for y_start, y_end in self.row_ranges():
            # Get columns containing an item
            nb_used_cells += sum(
                1
                for col in self.col_cells
                if any(
                    min(cell.y2, y_end) - max(cell.y1, y_start) >= 0.8 * cell.height for cell in col
                )
            )

        sparsity = 1 - nb_used_cells / (self.nb_columns * self.nb_rows)
        # Reward moderate sparsity and penalize both dense paragraphs and overly empty grids.
        return max(0.0, 1 - abs(sparsity - 0.35) / 0.35)

    def column_presence_ratios(self) -> list[float]:
        """
        Fraction of inferred rows containing content for each column.
        :return: list of ratios
        """
        if self.nb_rows == 0 or self.nb_columns == 0:
            return [0.0] * self.nb_columns

        return [
            sum(row[col_idx] for row in self.occupancy_matrix) / self.nb_rows
            for col_idx in range(self.nb_columns)
        ]

    def network_connectivity_score(self) -> float:
        """
        Measure how many occupied cells belong to a coherent row/column network.
        :return: score between 0 and 1
        """
        occupied = [
            (row_idx, col_idx)
            for row_idx, row in enumerate(self.occupancy_matrix)
            for col_idx, val in enumerate(row)
            if val
        ]
        if not occupied:
            return 0.0

        row_degrees = [sum(row) for row in self.occupancy_matrix]
        col_degrees = [
            sum(row[col_idx] for row in self.occupancy_matrix) for col_idx in range(self.nb_columns)
        ]
        coherent = [
            1
            for row_idx, col_idx in occupied
            if row_degrees[row_idx] >= 2 and col_degrees[col_idx] >= 2
        ]
        return len(coherent) / len(occupied)

    def row_pattern_consistency_score(self) -> float:
        """
        Measure how stable occupancy masks are from row to row.
        :return: score between 0 and 1
        """
        masks = [tuple(row) for row in self.occupancy_matrix if any(row)]
        if not masks:
            return 0.0

        dominant_ratio = Counter(masks).most_common(1)[0][1] / len(masks)
        if len(masks) > 2 and Counter(masks)[masks[0]] == 1:
            tail_ratio = Counter(masks[1:]).most_common(1)[0][1] / (len(masks) - 1)
            dominant_ratio = max(dominant_ratio, tail_ratio)

        return dominant_ratio

    def _alignment_summary(self) -> tuple[list[float | None], float, float]:
        column_scores = [self._col_alignment_score(col) for col in self.col_cells]
        valid_scores = [score for score in column_scores if score is not None]
        mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
        min_score = min(valid_scores, default=0.0)
        return column_scores, mean_score, min_score

    def table_score(self) -> float:
        """
        Compute weighted table confidence score.
        :return: score between 0 and 1
        """
        _, mean_column_alignment, min_column_alignment = self._alignment_summary()
        score = (
            0.20 * mean_column_alignment
            + 0.15 * min_column_alignment
            + 0.15 * self.content_spacing_consistency()
            + 0.20 * self.network_connectivity_score()
            + 0.05 * self.row_pattern_consistency_score()
            + 0.05 * self.sparsity_score()
            - 0.025 * self.full_text_score()
        )
        return max(0.0, min(1.0, score))

    def hard_reject_reason(self) -> str | None:
        """
        Identify obvious non-table patterns before scoring.
        :return: rejection reason or None
        """
        reason = None
        if self.nb_rows < 2:
            reason = "insufficient_rows"
        elif self.nb_columns < 2:
            reason = "insufficient_columns"
        else:
            presence_ratios = self.column_presence_ratios()
            network_connectivity = self.network_connectivity_score()
            _, mean_column_alignment, _ = self._alignment_summary()
            spacing_consistency = self.content_spacing_consistency()

            if sum(ratio >= 0.5 for ratio in presence_ratios) < 2:
                reason = "insufficient_column_presence"
            elif network_connectivity < 0.35:
                reason = "weak_network_connectivity"
            elif mean_column_alignment < 0.35 and spacing_consistency < 0.35:
                reason = "weak_alignment_and_spacing"

        return reason

    def is_structured(self) -> bool:
        """
        Assess whether the section behaves like a table
        :return: True if the table is structured, False otherwise
        """
        if self.hard_reject_reason() is not None:
            return False
        return self.table_score() >= 0.425

    @property
    def characteristics(self) -> dict:
        """
        Get the characteristics of the table
        :return: dictionary of characteristics
        """
        _, mean_column_alignment, min_column_alignment = self._alignment_summary()
        return {
            "mean_column_alignment": mean_column_alignment,
            "min_column_alignment": min_column_alignment,
            "content_spacing_consistency": self.content_spacing_consistency(),
            "column_presence_ratios": self.column_presence_ratios(),
            "network_connectivity_score": self.network_connectivity_score(),
            "row_pattern_consistency_score": self.row_pattern_consistency_score(),
            "sparsity_score": self.sparsity_score(),
            "full_text_score": self.full_text_score(),
            "table_score": self.table_score(),
            "hard_reject_reason": self.hard_reject_reason(),
        }
