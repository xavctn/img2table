from collections import Counter
from typing import TYPE_CHECKING

from img2table.tables.processing.borderless_tables_v2._model import identify_merged_rows

if TYPE_CHECKING:
    from img2table.tables.processing.borderless_tables_v2.tables.filter.model import (
        StructuredSection,
    )


def _cell_content_map(section: "StructuredSection") -> list[list[list]]:
    """
    Assign each row item to the column containing its x-center.
    :param section: The structured section to compute the metric for.
    :return: row/column inferred cell contents
    """
    # Initialize matrix
    matrix = [[[] for _ in range(section.nb_columns)] for _ in section.row_ranges()]
    if section.nb_columns == 0:
        return matrix

    # Identify cells by rows
    row_cells = []
    for y_start, y_end in section.row_ranges():
        row_items = [
            cell
            for cell in section.items
            if min(cell.y2, y_end) - max(cell.y1, y_start) >= 0.5 * cell.height
        ]
        row_cells.append(sorted(row_items, key=lambda cell: (cell.x1, cell.y1)))

    # Compute occupancy matrix
    for row_idx, row_items in enumerate(row_cells):
        for cell in row_items:
            center = (cell.x1 + cell.x2) / 2
            for col_idx, (start, end) in enumerate(section.cols):
                if start <= center <= end:
                    matrix[row_idx][col_idx].append(cell)
                    break

    return matrix


def _occupancy_matrix(cell_content_map: list[list[list]]) -> list[list[bool]]:
    """
    Convert inferred cell contents into an occupancy matrix.
    :param cell_content_map: row/column inferred cell contents
    :return: row/column occupancy matrix
    """
    return [[bool(cell_items) for cell_items in row] for row in cell_content_map]


def column_presence_ratios(
    section: "StructuredSection", occupancy_matrix: list[list[bool]]
) -> list[float]:
    """
    Fraction of inferred rows containing content for each column.
    :param section: The structured section to compute the metric for.
    :param occupancy_matrix: Matrix indicating which cells are occupied.
    :return: list of ratios
    """
    if section.nb_rows == 0 or section.nb_columns == 0:
        return [0.0] * section.nb_columns

    return [
        sum(row[col_idx] for row in occupancy_matrix) / section.nb_rows
        for col_idx in range(section.nb_columns)
    ]


def network_connectivity_score(
    section: "StructuredSection", occupancy_matrix: list[list[bool]]
) -> float:
    """
    Measure how many occupied cells belong to a coherent row/column network.
    :param section: The structured section to compute the metric for.
    :param occupancy_matrix: Matrix indicating which cells are occupied.
    :return: score between 0 and 1
    """
    occupied = [
        (row_idx, col_idx)
        for row_idx, row in enumerate(occupancy_matrix)
        for col_idx, val in enumerate(row)
        if val
    ]
    if not occupied:
        return 0.0

    row_degrees = [sum(row) for row in occupancy_matrix]
    col_degrees = [
        sum(row[col_idx] for row in occupancy_matrix) for col_idx in range(section.nb_columns)
    ]
    coherent = [
        1
        for row_idx, col_idx in occupied
        if row_degrees[row_idx] >= 2 and col_degrees[col_idx] >= 2
    ]
    return len(coherent) / len(occupied)


def row_pattern_consistency_score(occupancy_matrix: list[list[bool]]) -> float:
    """
    Measure how stable occupancy masks are from row to row.
    :param occupancy_matrix: Matrix indicating which cells are occupied.
    :return: score between 0 and 1
    """
    masks = [tuple(row) for row in occupancy_matrix if any(row)]
    if not masks:
        return 0.0

    dominant_ratio = Counter(masks).most_common(1)[0][1] / len(masks)
    if len(masks) > 2 and Counter(masks)[masks[0]] == 1:
        tail_ratio = Counter(masks[1:]).most_common(1)[0][1] / (len(masks) - 1)
        dominant_ratio = max(dominant_ratio, tail_ratio)

    return dominant_ratio


def nonsense_cell_ratio(section: "StructuredSection", cell_content_map: list[list[list]]) -> float:
    """
    Measure the share of occupied inferred cells whose content looks unlike regular text.
    :param section: The structured section to compute the metric for.
    :param cell_content_map: row/column inferred cell contents
    :return: score between 0 and 1
    """
    suspicious_cells, occupied_cells = 0, 0
    row_ranges = section.row_ranges()

    for row_idx, row in enumerate(cell_content_map):
        y_start, y_end = row_ranges[row_idx]
        cell_height = max(1, y_end - y_start)

        for col_idx, cell_items in enumerate(row):
            if not cell_items:
                continue

            occupied_cells += 1
            col_start, col_end = section.cols[col_idx]
            cell_width = max(1, col_end - col_start)

            merged_rows = identify_merged_rows(cnts=cell_items)
            vertical_span = (
                max(item.y2 for item in cell_items) - min(item.y1 for item in cell_items)
            ) / cell_height
            small_fragment_ratio = sum(
                item.width <= 1.5 * section.char_length and item.height <= 1.25 * section.char_length
                for item in cell_items
            ) / len(cell_items)
            bbox_fill_ratio = min(
                sum(item.area for item in cell_items),
                cell_width * cell_height,
            ) / (cell_width * cell_height)

            if len(cell_items) < 2:
                continue
            if len(merged_rows) < 2 and vertical_span < 0.5:
                continue
            if small_fragment_ratio < 0.3 and bbox_fill_ratio < 0.3:
                continue

            suspicious_cells += 1

    return suspicious_cells / occupied_cells if occupied_cells else 0.0


def compute_content_layout_metrics(
    section: "StructuredSection",
) -> tuple[list[float], float, float, float]:
    """
    Compute content layout / consistency metrics.
    :param section: The structured section to compute metrics for.
    :return: A tuple of the mean and minimum alignment score.
    """
    # Compute inferred cell contents and occupancy matrix
    cell_content_map = _cell_content_map(section=section)
    occupancy_matrix = _occupancy_matrix(cell_content_map=cell_content_map)

    # Compute content layout metrics
    presence_ratios = column_presence_ratios(section=section, occupancy_matrix=occupancy_matrix)
    connectivity = network_connectivity_score(section=section, occupancy_matrix=occupancy_matrix)
    row_pattern_consistency = row_pattern_consistency_score(occupancy_matrix=occupancy_matrix)
    cell_nonsense_ratio = nonsense_cell_ratio(section=section, cell_content_map=cell_content_map)

    return presence_ratios, connectivity, row_pattern_consistency, cell_nonsense_ratio
