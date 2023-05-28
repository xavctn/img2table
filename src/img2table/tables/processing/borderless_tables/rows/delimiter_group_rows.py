# coding: utf-8
from typing import List, Optional, Tuple

import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import TableRow, DelimiterGroup


def get_delimiter_group_row_separation(delimiter_group: DelimiterGroup) -> Optional[float]:
    """
    Identify median row separation between elements of the delimiter group
    :param delimiter_group: column delimiters group
    :return: median row separation in pixels
    """
    # Create dataframe with delimiter group elements
    list_elements = [{"id": idx, "x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2}
                     for idx, el in enumerate(delimiter_group.elements)]
    df_elements = pl.LazyFrame(data=list_elements)

    # Cross join to get corresponding elements and filter on elements that corresponds horizontally
    df_h_elms = (df_elements.join(df_elements, how='cross')
                 .filter(pl.col('id') != pl.col('id_right'))
                 .filter(pl.min([pl.col('x2'), pl.col('x2_right')])
                         - pl.max([pl.col('x1'), pl.col('x1_right')]) > 0)
                 )

    # Get element which is directly below
    df_elms_below = (df_h_elms.filter(pl.col('y1') < pl.col('y1_right'))
                     .sort(['id', 'y1_right'])
                     .with_columns(pl.lit(1).alias('ones'))
                     .with_columns(pl.col('ones').cumsum().over(["id"]).alias('rk'))
                     .filter(pl.col('rk') == 1)
                     )

    if df_elms_below.collect(streaming=True).height == 0:
        return None

    # Compute median vertical distance between elements
    median_v_dist = (df_elms_below.with_columns(((pl.col('y1_right') + pl.col('y2_right')
                                                  - pl.col('y1') - pl.col('y2')) / 2).abs().alias('y_diff'))
                     .select(pl.median('y_diff'))
                     .collect(streaming=True)
                     .to_dicts()
                     .pop()
                     .get('y_diff')
                     )

    return median_v_dist


def identify_rows(elements: List[Cell], ref_size: int) -> List[TableRow]:
    """
    Identify rows from Cell elements
    :param elements: list of cells
    :param ref_size: reference distance between two line centers
    :return: list of table rows
    """
    if len(elements) == 0:
        return []

    elements = sorted(elements, key=lambda c: c.y1 + c.y2)

    # Group elements in rows
    seq = iter(elements)
    tb_lines = [TableRow(cells=[next(seq)])]
    for cell in seq:
        if (cell.y1 + cell.y2) / 2 - tb_lines[-1].v_center > ref_size:
            tb_lines.append(TableRow(cells=[]))
        tb_lines[-1].add(cell)

    # Remove overlapping rows
    dedup_lines = list()
    for line in tb_lines:
        # Get number of overlapping rows
        overlap_lines = [l for l in tb_lines if line.overlaps(l) and not line == l]

        if len(overlap_lines) <= 1:
            dedup_lines.append(line)

    # Merge rows that corresponds
    merged_lines = [[l for l in dedup_lines if line.overlaps(l)] for line in dedup_lines]
    merged_lines = [line.pop() if len(line) == 1 else line[0].merge(line[1]) for line in merged_lines]

    return list(set(merged_lines))


def identify_delimiter_group_rows(delimiter_group: DelimiterGroup) -> Tuple[List[TableRow], float]:
    """
    Identify list of rows corresponding to the delimiter group
    :param delimiter_group: column delimiters group
    :return: list of rows corresponding to the delimiter group
    """
    # Identify median row separation between elements of the delimiter group
    group_median_row_sep = get_delimiter_group_row_separation(delimiter_group=delimiter_group)

    if group_median_row_sep:
        # Identify rows
        group_lines = identify_rows(elements=delimiter_group.elements,
                                    ref_size=int(group_median_row_sep // 3))
        return group_lines, group_median_row_sep

    return [], group_median_row_sep
