from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING

from img2table.tables.borderless.types import (
    LayoutRegion,
    identify_merged_rows,
)
from img2table.tables.common import cluster_items

if TYPE_CHECKING:
    from img2table.tables.types import Cell, Line


@dataclass
class VerticalWhitespace:
    x1: int
    x2: int
    y1: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class ColumnDelimiters:
    x_min: int
    x_max: int
    width: int
    vertical_ws: list[VerticalWhitespace] = field(default_factory=list)

    @property
    def y1(self) -> int:
        return max((ws.y1 for ws in self.vertical_ws), default=0)

    @property
    def y2(self) -> int:
        return min((ws.y2 for ws in self.vertical_ws), default=0)

    @property
    def relevant(self) -> bool:
        # At least two columns, at most 3
        if len(self.vertical_ws) == 0 or len(self.vertical_ws) > 2:
            return False

        # Get column widths
        col_widths = [
            nxt.x1 - prv.x2
            for prv, nxt in pairwise(
                sorted(
                    [
                        *self.vertical_ws,
                        VerticalWhitespace(x1=self.x_min, x2=self.x_min, y1=self.y1, y2=self.y2),
                        VerticalWhitespace(x1=self.x_max, x2=self.x_max, y1=self.y1, y2=self.y2),
                    ],
                    key=lambda ws: ws.x1,
                )
            )
        ]
        if max(col_widths) / min(col_widths) <= 1.25:
            return True

        # If only two columns, we can check if one column is about double the size of the other
        return len(self.vertical_ws) == 1 and 1.9 <= max(col_widths) / min(col_widths) <= 2.1

    @cached_property
    def columns(self) -> list[tuple[int, int]]:
        return list(
            pairwise(sorted([0, self.width] + [(ws.x1 + ws.x2) // 2 for ws in self.vertical_ws]))
        )

    def as_layout(self, contours: list[Cell]) -> list[LayoutRegion]:
        return [
            LayoutRegion.build(
                x1=start,
                y1=self.y1,
                x2=end,
                y2=self.y2,
                contours=contours,
            )
            for start, end in self.columns
        ]


def cut_ws_by_lines(
    ws: VerticalWhitespace, lines: list[Line], min_height: float
) -> list[VerticalWhitespace]:
    """
    Enforce rule that vertical whitespaces can not cross an horizontal line
    :param ws: vertical whitespace
    :param lines: list of horizontal image lines sorted by y coordinate
    :param min_height: minimum height for an eligible whitespace
    :return: list of updated whitespaces
    """
    # Create cut whitespaces
    cut_whitespaces: list[VerticalWhitespace] = []
    current_y = ws.y1

    for line in lines:
        # Ignore lines above the current segment
        if line.y1 <= current_y:
            continue
        # Stop once lines are below the whitespace
        if line.y1 >= ws.y2:
            break
        # Ignore lines that do not sufficiently overlap the whitespace
        if min(ws.x2, line.x2) - max(ws.x1, line.x1) < 0.5 * ws.width:
            continue

        if line.y1 - current_y >= min_height:
            cut_whitespaces.append(VerticalWhitespace(x1=ws.x1, y1=current_y, x2=ws.x2, y2=line.y1))
        current_y = line.y1

    # If no lines cross the whitespace, keep it unchanged
    if current_y == ws.y1:
        return [ws]

    # Add last whitespace segment if relevant
    if ws.y2 - current_y >= min_height:
        cut_whitespaces.append(VerticalWhitespace(x1=ws.x1, y1=current_y, x2=ws.x2, y2=ws.y2))

    return cut_whitespaces


def identify_vertical_delimiters(
    contours: list[Cell], lines: list[Line], min_width: float, width: int
) -> list[VerticalWhitespace]:
    """
    Identify possible vertical delimiters in page
    :param contours: list of image contours
    :param lines: list of image lines
    :param min_width: minimum width for a whitespace to be considered
    :param width: image width
    :return: list of whitespaces
    """
    if len(contours) == 0:
        return []

    # Compute minimum height
    min_height = (max(cnt.y2 for cnt in contours) - min(cnt.y1 for cnt in contours)) / 2
    h_lines = sorted((line for line in lines if line.horizontal), key=lambda line: line.y1)

    vertical_ws: list[VerticalWhitespace] = []
    current_ws: list[VerticalWhitespace] = []
    for row in identify_merged_rows(cnts=contours):
        # Compute row whitespaces
        row_ws = row.compute_whitespaces(min_width=min_width, x_min=0, x_max=width)

        # Check if row whitespaces correspond to current whitespaces
        updated_ws: list[VerticalWhitespace] = []
        used_ws = [False] * len(current_ws)
        current_idx = 0
        for r_ws in row_ws:
            used_row = False

            # Skip current whitespaces ending before the row whitespace starts
            while current_idx < len(current_ws) and current_ws[current_idx].x2 < r_ws.start:
                current_idx += 1

            # Identify matching current ws
            scan_idx = current_idx
            while scan_idx < len(current_ws) and current_ws[scan_idx].x1 <= r_ws.end:
                ws = current_ws[scan_idx]
                if min(r_ws.end, ws.x2) - max(r_ws.start, ws.x1) >= min_width:
                    # Add updated vertical ws
                    updated_ws.append(
                        VerticalWhitespace(
                            x1=max(r_ws.start, ws.x1), y1=ws.y1, x2=min(r_ws.end, ws.x2), y2=row.y2
                        )
                    )
                    used_ws[scan_idx] = True
                    used_row = True
                scan_idx += 1

            # If the row has not been used, create new vertical whitespace
            if not used_row:
                updated_ws.append(
                    VerticalWhitespace(x1=r_ws.start, y1=row.y1, x2=r_ws.end, y2=row.y2)
                )

        # Identify relevant vertical ws from the unused ones
        vertical_ws += [
            cut_ws
            for idx, ws in enumerate(current_ws)
            for cut_ws in cut_ws_by_lines(ws=ws, lines=h_lines, min_height=min_height)
            if not used_ws[idx] and ws.x1 > 0 and ws.x2 < width and cut_ws.height >= min_height
        ]

        # Update current whitespaces
        current_ws = updated_ws

    # Purge current ws
    vertical_ws += [
        cut_ws
        for ws in current_ws
        for cut_ws in cut_ws_by_lines(ws=ws, lines=h_lines, min_height=min_height)
        if ws.x1 > 0 and ws.x2 < width and cut_ws.height >= min_height
    ]

    return vertical_ws


def vertical_ws_matches(ws1: VerticalWhitespace, ws2: VerticalWhitespace) -> bool:
    """
    Assess if whitespaces top or bottom bound matches
    :param ws1: first vertical whitespaces
    :param ws2: second vertical whitespaces
    :return: boolean indicating if whitespaces top or bottom bound matches
    """
    # Top matches
    return min(abs(ws1.y1 - ws2.y1), abs(ws1.y2 - ws2.y2)) / max(ws1.height, ws2.height) <= 0.05


def identify_column_regions(
    vertical_ws: list[VerticalWhitespace], x_min: int, x_max: int, width: int
) -> list[ColumnDelimiters]:
    """
    Identify column sections from vertical delimiters
    :param vertical_ws: list of vertical whitespaces
    :param x_min: minimum x coordinate
    :param x_max: maximum x coordinate
    :param width: image width
    :return: list of column sections
    """
    if len(vertical_ws) == 0:
        return []

    # Create clusters of matching whitespaces and assign to column delimiters if relevant
    return [
        col
        for cl in cluster_items(items=vertical_ws, clustering_func=vertical_ws_matches)
        if (col := ColumnDelimiters(x_min=x_min, x_max=x_max, width=width, vertical_ws=cl)).relevant
    ]


def create_all_layout_areas(
    column_dels: list[ColumnDelimiters],
    contours: list[Cell],
    y_min: int,
    y_max: int,
    width: int,
) -> list[LayoutRegion]:
    """
    Identify layout in image
    :param contours: list of image contours
    :param contours: list of image contours
    :param y_min: minimum y coordinate
    :param y_max: maximum y coordinate
    :param width: image width
    :return: list of layout areas from image
    """
    # Identify vertical spans not covered by column delimiters
    layout_areas, current_y = [], y_min
    for col in sorted(column_dels, key=lambda col: col.y1):
        # Check gap width
        gap = col.y1 - current_y
        if gap > 0:
            layout_areas.append(
                LayoutRegion.build(x1=0, x2=width, y1=current_y, y2=col.y1, contours=contours)
            )
        current_y = max(current_y, col.y2)

    # Add last area if relevant
    if y_max - current_y > 0:
        layout_areas.append(
            LayoutRegion.build(x1=0, x2=width, y1=current_y, y2=y_max, contours=contours)
        )

    # Add areas from column delimters
    layout_areas += [area for col in column_dels for area in col.as_layout(contours=contours)]

    return sorted(
        [area for area in layout_areas if len(area.contours) > 0],
        key=lambda area: (area.y1, area.x1),
    )


def identify_layout(
    contours: list[Cell], lines: list[Line], min_width: float, width: int
) -> list[LayoutRegion]:
    """
    Identify layout in image
    :param contours: list of image contours
    :param lines: list of image lines
    :param min_width: minimum width for a whitespace to be considered
    :param width: image width
    :return: list of layout areas from image
    """
    # Identify vertical delimiters
    vertical_ws = identify_vertical_delimiters(
        contours=contours, lines=lines, min_width=min_width, width=width
    )

    # Get column delimiters from vertical whitespaces
    y_min, y_max = (
        min((cnt.y1 for cnt in contours), default=0),
        max((cnt.y2 for cnt in contours), default=0),
    )
    column_dels = identify_column_regions(
        vertical_ws=vertical_ws,
        x_min=min((cnt.x1 for cnt in contours), default=0),
        x_max=max((cnt.x2 for cnt in contours), default=0),
        width=width,
    )

    if len(column_dels) == 0:
        return [LayoutRegion(x1=0, x2=width, y1=y_min, y2=y_max, contours=contours)]

    return create_all_layout_areas(
        column_dels=column_dels, contours=contours, y_min=y_min, y_max=y_max, width=width
    )
