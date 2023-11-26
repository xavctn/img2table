# coding: utf-8
from typing import List, Optional, Tuple, Set

import polars as pl

from img2table.tables import find_components
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import TableRow, DelimiterGroup


def get_delimiter_group_row_separation(delimiter_group: DelimiterGroup) -> Optional[float]:
    """
    Identify median row separation between elements of the delimiter group
    :param delimiter_group: column delimiters group
    :return: median row separation in pixels
    """
    if len(delimiter_group.elements) == 0:
        return None

    # Create dataframe with delimiter group elements
    list_elements = [{"id": idx, "x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2}
                     for idx, el in enumerate(delimiter_group.elements)]
    df_elements = pl.LazyFrame(data=list_elements)

    # Cross join to get corresponding elements and filter on elements that corresponds horizontally
    df_h_elms = (df_elements.join(df_elements, how='cross')
                 .filter(pl.col('id') != pl.col('id_right'))
                 .filter(pl.min_horizontal(['x2', 'x2_right']) - pl.max_horizontal(['x1', 'x1_right']) > 0)
                 )

    # Get element which is directly below
    df_elms_below = (df_h_elms.filter(pl.col('y1') < pl.col('y1_right'))
                     .sort(['id', 'y1_right'])
                     .with_columns(pl.lit(1).alias('ones'))
                     .with_columns(pl.col('ones').cum_sum().over(["id"]).alias('rk'))
                     .filter(pl.col('rk') == 1)
                     )

    if df_elms_below.collect().height == 0:
        return None

    # Compute median vertical distance between elements
    median_v_dist = (df_elms_below.with_columns(((pl.col('y1_right') + pl.col('y2_right')
                                                  - pl.col('y1') - pl.col('y2')) / 2).abs().alias('y_diff'))
                     .select(pl.median('y_diff'))
                     .collect()
                     .to_dicts()
                     .pop()
                     .get('y_diff')
                     )

    return median_v_dist


def identify_aligned_elements(df_elements: pl.DataFrame, ref_size: int) -> List[Set[int]]:
    """
    Identify groups of elements that are vertically aligned
    :param df_elements: dataframe of elements
    :param ref_size: reference distance between two line centers
    :return: list of element sets which represent aligned elements
    """
    # Cross join elements and filter on aligned elements
    df_cross = (df_elements.join(df_elements, how="cross")
                .filter(((pl.col('y1') + pl.col('y2') - pl.col('y1_right') - pl.col('y2_right')) / 2).abs() <= ref_size)
                .filter(pl.max_horizontal(pl.col('height'), pl.col('height_right'))
                        / pl.min_horizontal(pl.col('height'), pl.col('height_right')) <= 2)
                .select(pl.struct(pl.min_horizontal(pl.col('idx'), pl.col('idx_right')).alias('idx_1'),
                                  pl.max_horizontal(pl.col('idx'), pl.col('idx_right')).alias('idx_2')).alias('idxs')
                        )
                .unique()
                .unnest("idxs")
                )
    aligned_elements = [{row.get('idx_1'), row.get('idx_2')} for row in df_cross.to_dicts()]

    # Create cluster of aligned elements
    clusters = find_components(edges=aligned_elements)

    return clusters


def identify_overlapping_row_clusters(df_elements: pl.DataFrame, clusters: List[Set[int]]) -> List[Set[int]]:
    """
    Identify rows that overlap with each other and group them in cluster
    :param df_elements: dataframe of elements
    :param clusters: list of element sets which represent aligned elements/ rows
    :return: list of clusters sets which represent aligned rows
    """
    clusters_dict = {el_idx: cl_idx for cl_idx, cl in enumerate(clusters) for el_idx in cl}
    df_elements = df_elements.with_columns(pl.col('idx').map_elements(clusters_dict.get).alias("cl_idx"))

    df_clusters = (df_elements.group_by("cl_idx")
                   .agg(pl.col("x1").min().alias('x1'),
                        pl.col("y1").min().alias('y1'),
                        pl.col("x2").max().alias('x2'),
                        pl.col("y2").max().alias('y2'))
                   .with_columns((pl.col("y2") - pl.col("y1")).alias('height'))
                   )

    df_cross = (df_clusters.join(df_clusters, how='cross')
                .with_columns((pl.min_horizontal(pl.col("y2"), pl.col("y2_right"))
                               - pl.max_horizontal(pl.col("y1"), pl.col("y1_right"))).alias('overlap')
                             )
                .filter(pl.col('overlap') / pl.min_horizontal(pl.col("height"), pl.col("height_right")) >= 0.5)
                .select(pl.struct(pl.min_horizontal(pl.col('cl_idx'), pl.col('cl_idx_right')).alias('idx_1'),
                                  pl.max_horizontal(pl.col('cl_idx'), pl.col('cl_idx_right')).alias('idx_2')).alias('idxs')
                        )
                .unique()
                .unnest("idxs")
                )
    aligned_rows = [{row.get('idx_1'), row.get('idx_2')} for row in df_cross.to_dicts()]

    # Create cluster of aligned rows
    row_clusters = find_components(edges=aligned_rows)

    return row_clusters


def not_overlapping_rows(tb_row_1: TableRow, tb_row_2: TableRow) -> bool:
    """
    Identify if two TableRow objects do not overlap vertically
    :param tb_row_1: first TableRow object
    :param tb_row_2: second TableRow object
    :return: boolean indicating if both TableRow objects do not overlap vertically
    """
    # Compute overlap
    overlap = min(tb_row_1.y2, tb_row_2.y2) - max(tb_row_1.y1, tb_row_2.y1)
    return overlap / min(tb_row_1.height, tb_row_2.height) <= 0.1


def score_row_group(row_group: List[TableRow], height: int, max_elements: int) -> float:
    """
    Score row group pertinence
    :param row_group: group of TableRow objects
    :param height: reference height
    :param max_elements: reference number of elements/cells that can be included in a row group
    :return: scoring of the row group
    """
    # Get y coverage of row group
    y_total = sum([r.height for r in row_group])
    y_overlap = sum([max(0, min(r_1.y2, r_2.y2) - max(r_1.y1, r_2.y1)) for r_1, r_2 in zip(row_group, row_group[1:])])
    y_coverage = y_total - y_overlap

    # Score row group
    return (sum([len(r.cells) for r in row_group]) / max_elements) * (y_coverage / height)


def get_rows_from_overlapping_cluster(row_cluster: List[TableRow]) -> List[TableRow]:
    """
    Identify relevant rows from a cluster of vertically overlapping rows
    :param row_cluster: cluster of vertically overlapping TableRow objects
    :return: relevant rows from a cluster of vertically overlapping rows
    """
    # Get height of row cluster
    ref_height = max([r.y2 for r in row_cluster]) - min([r.y1 for r in row_cluster])

    # Get groups of distinct rows
    seq = iter(row_cluster)
    distinct_rows_clusters = [[next(seq)]]
    for row in seq:
        for idx, cl in enumerate(distinct_rows_clusters):
            if all([not_overlapping_rows(tb_row_1=row, tb_row_2=r) for r in cl]):
                distinct_rows_clusters[idx].append(row)
        distinct_rows_clusters.append([row])

    # Get maximum number of elements possible in a row cluster
    max_elements = max([sum([len(r.cells) for r in cl]) for cl in distinct_rows_clusters])

    # Sort elements by score
    scored_elements = sorted(distinct_rows_clusters,
                             key=lambda gp: score_row_group(row_group=gp,
                                                            height=ref_height,
                                                            max_elements=max_elements)
                             )

    # Get cluster of rows with the largest score
    return scored_elements.pop()


def identify_rows(elements: List[Cell], ref_size: int) -> List[TableRow]:
    """
    Identify rows from Cell elements
    :param elements: list of cells
    :param ref_size: reference distance between two line centers
    :return: list of table rows
    """
    if len(elements) == 0:
        return []

    # Create dataframe with elements
    df_elements = (pl.DataFrame([{"idx": idx, "x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2}
                                 for idx, el in enumerate(elements)])
                   .with_columns((pl.col('y2') - pl.col('y1')).alias('height'))
                   )

    # Identify clusters of elements that represent rows
    element_clusters = identify_aligned_elements(df_elements=df_elements,
                                                 ref_size=ref_size)

    # Identify overlapping rows
    row_clusters = identify_overlapping_row_clusters(df_elements=df_elements,
                                                     clusters=element_clusters)

    overlap_row_clusters = [[TableRow(cells=[elements[idx] for idx in element_clusters[cl_idx]]) for cl_idx in row_cl]
                            for row_cl in row_clusters]

    # Get relevant rows in each cluster
    relevant_rows = [row for cl in overlap_row_clusters
                     for row in get_rows_from_overlapping_cluster(cl)]

    # Check for overlapping rows
    seq = iter(sorted(relevant_rows, key=lambda r: r.y1 + r.y2))
    final_rows = [next(seq)]
    for row in seq:
        if row.overlaps(final_rows[-1]):
            final_rows[-1].merge(row)
        else:
            final_rows.append(row)

    return final_rows


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

        # Adjust height of first / last row
        if group_lines:
            group_lines[0].set_y_top(delimiter_group.y1)
            group_lines[-1].set_y_bottom(delimiter_group.y2)

        return group_lines, group_median_row_sep

    return [], group_median_row_sep
