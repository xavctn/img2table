from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

from img2table.tables.types import Cell

if TYPE_CHECKING:
    from img2table.tables.types import Table


def get_title_areas(
    contours: list[Cell], tables: list[Table], median_line_sep: float
) -> list[Table]:
    """
    Identify title areas for each table
    :param contours: list of contours
    :param tables: list of tables
    :param median_line_sep: median line separation
    :return: list of tables with title areas set
    """
    contours = sorted(contours, key=lambda c: (c.y1, c.x1))

    final_tables = []
    if len(tables) == 0:
        return final_tables

    # Identify clusters of vertically overlapping tables
    tables = sorted(tables, key=lambda t: (t.y1, t.x1))
    table_clusters: list[list[Table]] = [[]]
    for table in tables:
        if (current_cluster := table_clusters[-1]) and table.y1 > current_cluster[-1].y2:
            table_clusters.append([])
        table_clusters[-1].append(table)

    # Compute vertical ranges eligible for each cluster
    vertical_limits = [0, *[max(tb.y2 for tb in cluster) for cluster in table_clusters[:-1]]]

    for y_min, cluster in zip(vertical_limits, table_clusters, strict=True):
        sorted_cluster = sorted(cluster, key=lambda t: t.x1)

        # Compute horizontal bounds
        h_bounds = [
            0,
            *[round((prv.x2 + nxt.x1) / 2) for prv, nxt in pairwise(sorted_cluster)],
            10e6,
        ]

        for table, (x_min, x_max) in zip(sorted_cluster, pairwise(h_bounds), strict=True):
            # Get contours within the relevant area
            targeted_contours = []
            for cnt in contours:
                if cnt.y1 >= table.y1:
                    break
                if (
                    cnt.y1 >= y_min
                    and cnt.y2 < table.y1
                    and cnt.x1 >= max(x_min, table.x1)
                    and cnt.x2 <= min(x_max, table.x2)
                ):
                    targeted_contours.append(cnt)

            if len(targeted_contours) == 0:
                final_tables.append(table)
                continue

            # Keep contours within a "median_line_sep" of the closest one
            min_height = max(cnt.y1 for cnt in targeted_contours) - median_line_sep
            relevant_contours = [cnt for cnt in targeted_contours if cnt.y1 >= min_height]

            # Update table title area
            table.set_title_area(
                title_area=Cell(
                    x1=min(cnt.x1 for cnt in relevant_contours),
                    y1=min(cnt.y1 for cnt in relevant_contours),
                    x2=max(cnt.x2 for cnt in relevant_contours),
                    y2=max(cnt.y2 for cnt in relevant_contours),
                )
            )
            final_tables.append(table)

    return final_tables
