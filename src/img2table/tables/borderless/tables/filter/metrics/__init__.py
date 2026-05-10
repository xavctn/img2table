from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from img2table.tables.borderless.tables.filter.metrics.columns import (
    compute_columns_metrics,
)
from img2table.tables.borderless.tables.filter.metrics.content import (
    compute_content_layout_metrics,
)
from img2table.tables.borderless.tables.filter.metrics.misc import (
    full_text_score,
    sparsity_score,
)
from img2table.tables.borderless.tables.filter.metrics.rows import (
    content_spacing_consistency,
)

if TYPE_CHECKING:
    from img2table.tables.borderless.tables.filter.model import (
        StructuredSection,
    )


@dataclass
class TableMetrics:
    presence_ratios: list[float]
    network_connectivity: float
    mean_column_alignment: float
    min_column_alignment: float
    spacing_consistency: float
    row_pattern_consistency: float
    sparsity: float
    full_text: float

    def score(self) -> float:
        """
        Compute aggregated score of the table metrics
        :return: aggregated score
        """
        return (
            0.20 * self.mean_column_alignment
            + 0.15 * self.min_column_alignment
            + 0.15 * self.spacing_consistency
            + 0.20 * self.network_connectivity
            + 0.05 * self.row_pattern_consistency
            + 0.05 * self.sparsity
            - 0.025 * self.full_text
        )

    @classmethod
    def from_section(cls, section: StructuredSection) -> TableMetrics:
        # Compute metrics
        mean_column_alignment, min_column_alignment = compute_columns_metrics(section=section)
        spacing_consistency = content_spacing_consistency(section=section)
        (
            presence_ratios,
            connectivity,
            row_pattern_consistency,
        ) = compute_content_layout_metrics(section=section)
        full_text = full_text_score(section=section)
        sparsity = sparsity_score(section=section)

        return cls(
            presence_ratios=presence_ratios,
            network_connectivity=connectivity,
            mean_column_alignment=mean_column_alignment,
            min_column_alignment=min_column_alignment,
            spacing_consistency=spacing_consistency,
            row_pattern_consistency=row_pattern_consistency,
            sparsity=sparsity,
            full_text=full_text,
        )
