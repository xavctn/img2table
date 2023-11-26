# coding: utf-8

from typing import List, Union

import polars as pl

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import ImageSegment, DelimiterGroup


def get_whitespaces(segment: Union[ImageSegment, DelimiterGroup], vertical: bool = True, min_width: float = 0,
                    pct: float = 0.25) -> List[Cell]:
    """
    Identify whitespaces in segment
    :param segment: image segment
    :param vertical: boolean indicating if vertical or horizontal whitespaces are identified
    :param pct: minimum percentage of the segment height/width to account for a whitespace
    :param min_width: minimum width of the detected whitespaces
    :return: list of vertical or horizontal whitespaces as Cell objects
    """
    # Flip object coordinates in horizontal case
    if not vertical:
        flipped_elements = [Cell(x1=el.y1, y1=el.x1, x2=el.y2, y2=el.x2) for el in segment.elements]
        segment = ImageSegment(x1=segment.y1,
                               y1=segment.x1,
                               x2=segment.y2,
                               y2=segment.x2,
                               elements=flipped_elements)

    # Get min/max height of elements in segment
    y_min, y_max = min([el.y1 for el in segment.elements]), max([el.y2 for el in segment.elements])

    # Create dataframe containing elements
    df_elements = pl.concat(
        [pl.LazyFrame([{"x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2} for el in segment.elements]),
         pl.LazyFrame([{"x1": segment.x1, "y1": y, "x2": segment.x2, "y2": y} for y in [y_min, y_max]])]
    )

    # Get dataframe with relevant ranges
    df_x_ranges = (pl.concat([df_elements.select(pl.col('x1').alias('x')), df_elements.select(pl.col('x2').alias('x'))])
                   .unique()
                   .sort(by="x")
                   .select(pl.col('x').alias('x_min'), pl.col('x').shift(-1).alias('x_max'))
                   .filter(pl.col('x_max') - pl.col('x_min') >= min_width)
                   )

    # Get all elements within range and identify whitespaces
    df_elements_ranges = (
        df_x_ranges.join(df_elements, how='cross')
        .with_columns((pl.min_horizontal(pl.col('x_max'), pl.col('x2'))
                       - pl.max_horizontal(pl.col('x_min'), pl.col('x1')) > 0).alias('overlapping'))
        .filter(pl.col('overlapping'))
        .sort(by=["x_min", "x_max", pl.col("y1") + pl.col('y2')])
        .select(pl.col("x_min").alias('x1'),
                pl.col("x_max").alias("x2"),
                pl.col('y2').shift().over("x_min", 'x_max').alias('y1'),
                pl.col('y1').alias('y2')
                )
        .filter(pl.col('y2') - pl.col('y1') >= pct * (y_max - y_min))
        .sort(by=['y1', 'y2', 'x1'])
        .with_columns(((pl.col('x1') != pl.col('x2').shift())
                       | (pl.col('y1') != pl.col('y1').shift())
                       | (pl.col('y2') != pl.col('y2').shift())
                       ).cast(int).cum_sum().alias('ws_id')
                      )
        .group_by("ws_id")
        .agg(pl.col('x1').min().alias('x1'),
             pl.col('y1').min().alias('y1'),
             pl.col('x2').max().alias('x2'),
             pl.col('y2').max().alias('y2'))
        .drop("ws_id")
        .collect()
    )

    whitespaces = [Cell(**ws_dict) for ws_dict in df_elements_ranges.to_dicts()]

    # Flip object coordinates in horizontal case
    if not vertical:
        whitespaces = [Cell(x1=ws.y1, y1=ws.x1, x2=ws.y2, y2=ws.x2) for ws in whitespaces]

    return whitespaces


def adjacent_whitespaces(w_1: Cell, w_2: Cell) -> bool:
    """
    Identify if two whitespaces are adjacent
    :param w_1: first whitespace
    :param w_2: second whitespace
    :return: boolean indicating if two whitespaces are adjacent
    """
    x_coherent = len({w_1.x1, w_1.x2}.intersection({w_2.x1, w_2.x2})) > 0
    y_coherent = min(w_1.y2, w_2.y2) - max(w_1.y1, w_2.y1) > 0

    return x_coherent and y_coherent


def identify_coherent_v_whitespaces(v_whitespaces: List[Cell], char_length: float) -> List[Cell]:
    """
    From vertical whitespaces, identify the most relevant ones according to height, width and relative positions
    :param v_whitespaces: list of vertical whitespaces
    :param char_length: average character width in image
    :return: list of relevant vertical delimiters
    """
    # Create vertical delimiters groups
    v_groups = cluster_items(items=v_whitespaces,
                             clustering_func=adjacent_whitespaces)

    # Keep only delimiters that represent at least 75% of the height of their group
    v_delims = [d for gp in v_groups
                for d in [d for d in gp if d.height >= 0.75 * max([d.height for d in gp])]]

    # Group once again delimiters and keep only highest one in group
    v_delim_groups = cluster_items(items=v_delims,
                                   clustering_func=adjacent_whitespaces)

    # For each group, select a delimiter that has the largest height
    final_delims = list()
    for gp in v_delim_groups:
        if gp:
            # Get x center of group
            x_center = (min([d.x1 for d in gp]) + max([d.x2 for d in gp]))

            # Filter on tallest delimiters
            tallest_delimiters = [d for d in gp if d.height == max([d.height for d in gp])]

            # Add delimiter closest to the center of the group
            closest_del = sorted(tallest_delimiters, key=lambda d: abs(d.x1 + d.x2 - x_center)).pop(0)
            final_delims.append(closest_del)

    # Add all whitespaces of the largest height
    max_height_ws = [ws for ws in v_whitespaces if ws.height == max([w.height for w in v_whitespaces])]

    return list(set(final_delims + max_height_ws))


def get_relevant_vertical_whitespaces(segment: Union[ImageSegment, DelimiterGroup], char_length: float,
                                      pct: float = 0.25) -> List[Cell]:
    """
    Identify vertical whitespaces that can be column delimiters
    :param segment: image segment
    :param char_length: average character width in image
    :param pct: minimum percentage of the segment height for a vertical whitespace
    :return: list of vertical whitespaces that can be column delimiters
    """
    # Identify vertical whitespaces
    v_whitespaces = get_whitespaces(segment=segment,
                                    vertical=True,
                                    pct=pct,
                                    min_width=0.5 * char_length)

    # Identify relevant vertical whitespaces that can be column delimiters
    vertical_delims = identify_coherent_v_whitespaces(v_whitespaces=v_whitespaces,
                                                      char_length=char_length)

    return vertical_delims
