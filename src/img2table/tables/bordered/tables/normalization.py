from collections import defaultdict

import numpy as np

from img2table.tables.common import _cluster_values
from img2table.tables.types import Cell


def normalize_table_cells(cluster_cells: list[Cell], char_length: float) -> list[Cell]:
    """
    Normalize cells from table cells
    :param cluster_cells: list of cells that form a table
    :param char_length: average character length
    :return: list of normalized cells
    """
    # Get list of existing horizontal values and cluster them
    h_values = sorted({x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]})
    cluster_mapping = defaultdict(list)
    for cl_idx, value in zip(
        _cluster_values(values=h_values, median_gap_multiple=0.1, min_gap=char_length),
        h_values,
        strict=True,
    ):
        cluster_mapping[cl_idx].append(value)
    # Get horizontal delimiters from cluster mapping
    h_delims = sorted(round(np.mean(vals)) for vals in cluster_mapping.values())

    # Get list of existing vertical values and cluster them
    v_values = sorted({y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]})
    cluster_mapping = defaultdict(list)
    for cl_idx, value in zip(
        _cluster_values(values=v_values, median_gap_multiple=0.1, min_gap=0.5 * char_length),
        v_values,
        strict=True,
    ):
        cluster_mapping[cl_idx].append(value)
    # Get vertical delimiters from cluster mapping
    v_delims = sorted(round(np.mean(vals)) for vals in cluster_mapping.values())

    # Normalize all cells
    normalized_cells: list[Cell] = []
    for cell in cluster_cells:
        normalized_cell = Cell(
            x1=min(h_delims, key=lambda d: abs(d - cell.x1)),
            x2=min(h_delims, key=lambda d: abs(d - cell.x2)),
            y1=min(v_delims, key=lambda d: abs(d - cell.y1)),
            y2=min(v_delims, key=lambda d: abs(d - cell.y2)),
        )
        # Check if cell is not empty
        if normalized_cell.area > 0:
            normalized_cells.append(normalized_cell)

    return normalized_cells
