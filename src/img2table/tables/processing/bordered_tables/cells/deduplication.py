# coding: utf-8
from typing import List

import numpy as np
import polars as pl

from img2table.tables.objects.cell import Cell


def deduplicate_cells(df_cells: pl.LazyFrame) -> List[Cell]:
    """
    Deduplicate nested cells in order to keep the smallest ones
    :param df_cells: dataframe containing cells
    :return: cells after deduplication of the nested ones
    """
    # Get Cell objects
    cells = [Cell(**d) for d in df_cells.drop("index").collect().to_dicts()]

    # Create array of cell coverages
    d_dims = df_cells.select(pl.col("x2").max(), pl.col("y2").max()).collect().to_dicts().pop()
    coverage_array = np.ones((d_dims.get("y2"), d_dims.get("x2")), dtype=np.uint8)

    dedup_cells = list()
    for c in sorted(cells, key=lambda c: c.area):
        cropped = coverage_array[c.y1:c.y2, c.x1:c.x2]
        # If cell has at least 25% of its area not covered, add it
        if np.sum(cropped) >= 0.25 * c.area:
            dedup_cells.append(c)
            coverage_array[c.y1:c.y2, c.x1:c.x2] = 0

    return dedup_cells
