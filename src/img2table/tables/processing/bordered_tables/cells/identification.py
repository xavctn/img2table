# coding: utf-8
from typing import List

import polars as pl

from img2table.tables.objects.line import Line


def get_potential_cells_from_h_lines(df_h_lines: pl.LazyFrame) -> pl.LazyFrame:
    """
    Identify potential cells by matching corresponding horizontal lines
    :param df_h_lines: dataframe containing horizontal lines
    :return: dataframe containing potential cells
    """
    # Create copy of df_h_lines
    df_h_lines_cp = (df_h_lines.clone()
                     .rename({col: f"{col}_" for col in df_h_lines.columns})
                     )

    # Cross join with itself to get pairs of horizontal lines
    cross_h_lines = (df_h_lines.join(df_h_lines_cp, how='cross')
                     .filter(pl.col('y1') < pl.col('y1_'))
                     )

    # Compute horizontal correspondences between lines
    cross_h_lines = cross_h_lines.with_columns([
        (((pl.col('x1') - pl.col('x1_')) / pl.col('width')).abs() <= 0.02).alias("l_corresponds"),
        (((pl.col('x2') - pl.col('x2_')) / pl.col('width')).abs() <= 0.02).alias("r_corresponds"),
        (((pl.col('x1') <= pl.col('x1_')) & (pl.col('x1_') <= pl.col('x2')))
         | ((pl.col('x1_') <= pl.col('x1')) & (pl.col('x1') <= pl.col('x2_')))).alias('l_contained'),
        (((pl.col('x1') <= pl.col('x2_')) & (pl.col('x2_') <= pl.col('x2')))
         | ((pl.col('x1_') <= pl.col('x2')) & (pl.col('x2') <= pl.col('x2_')))).alias('r_contained')
    ])

    # Create condition on horizontal correspondence in order to use both lines and filter on relevant combinations
    matching_condition = ((pl.col('l_corresponds') | pl.col('l_contained'))
                          & (pl.col('r_corresponds') | pl.col('r_contained')))
    cross_h_lines = cross_h_lines.filter(matching_condition)

    # Create cell bbox from horizontal lines
    df_bbox = (cross_h_lines.select([pl.max([pl.col('x1'), pl.col('x1_')]).alias('x1_bbox'),
                                     pl.min([pl.col('x2'), pl.col('x2_')]).alias('x2_bbox'),
                                     pl.col('y1').alias("y1_bbox"),
                                     pl.col('y1_').alias('y2_bbox')]
                                    )
               .with_row_count(name="idx")
               )

    # Deduplicate on upper bound
    df_bbox = (df_bbox.sort(by=["x1_bbox", "x2_bbox", "y1_bbox", "y2_bbox"])
               .with_columns(pl.lit(1).alias('ones'))
               .with_columns(pl.col('ones').cumsum().over(["x1_bbox", "x2_bbox", "y1_bbox"]).alias('cell_rk'))
               .filter(pl.col('cell_rk') == 1)
               )

    # Deduplicate on lower bound
    df_bbox = (df_bbox.sort(by=["x1_bbox", "x2_bbox", "y2_bbox", "y1_bbox"], descending=[False, False, False, True])
               .with_columns(pl.lit(1).alias('ones'))
               .with_columns(pl.col('ones').cumsum().over(["x1_bbox", "x2_bbox", "y2_bbox"]).alias('cell_rk'))
               .filter(pl.col('cell_rk') == 1)
               .drop(['ones', 'cell_rk'])
               )

    return df_bbox


def get_cells_dataframe(horizontal_lines: List[Line], vertical_lines: List[Line]) -> pl.LazyFrame:
    """
    Create dataframe of all possible cells from horizontal and vertical lines
    :param horizontal_lines: list of horizontal lines
    :param vertical_lines: list of vertical lines
    :return: dataframe containing all cells
    """
    # Check for empty lines
    if len(horizontal_lines) * len(vertical_lines) == 0:
        return pl.DataFrame().lazy()

    # Create dataframe from horizontal and vertical lines
    df_h_lines = pl.LazyFrame(data=[l.dict for l in horizontal_lines])
    df_v_lines = pl.LazyFrame(data=[l.dict for l in vertical_lines])

    # Identify potential cells bboxes from horizontal lines
    df_bbox = get_potential_cells_from_h_lines(df_h_lines=df_h_lines)

    # Cross join with vertical lines
    df_bbox = df_bbox.with_columns(pl.max([(pl.col('x2_bbox') - pl.col('x1_bbox')) * 0.025,
                                           pl.lit(5.0)]).round(0).alias('h_margin')
                                   )
    df_bbox_v = df_bbox.join(df_v_lines, how='cross')

    # Check horizontal correspondence between cell and vertical lines
    horizontal_cond = ((pl.col("x1_bbox") - pl.col("h_margin") <= pl.col("x1"))
                       & (pl.col("x2_bbox") + pl.col("h_margin") >= pl.col("x1")))
    df_bbox_v = df_bbox_v.filter(horizontal_cond)

    # Check vertical overlapping
    df_bbox_v = (df_bbox_v.with_columns((pl.min([pl.col('y2'), pl.col('y2_bbox')])
                                         - pl.max([pl.col('y1'), pl.col('y1_bbox')])).alias('overlapping')
                                        )
                 .filter(pl.col('overlapping') / (pl.col('y2_bbox') - pl.col('y1_bbox')) >= 0.8)
                 )

    # Get all vertical delimiters by bbox
    df_bbox_delimiters = (df_bbox_v.sort(['idx', "x1_bbox", "x2_bbox", "y1_bbox", "y2_bbox", "x1"])
                          .groupby(['idx', "x1_bbox", "x2_bbox", "y1_bbox", "y2_bbox"])
                          .agg(pl.col('x1').alias('dels'))
                          .filter(pl.col("dels").arr.lengths() >= 2)
                          )

    # Create new cells based on vertical delimiters
    df_cells = (df_bbox_delimiters.explode("dels")
                .with_columns([pl.col('dels').shift(1).over(pl.col('idx')).alias("x1_bbox"),
                               pl.col('dels').alias("x2_bbox")])
                .filter(pl.col('x1_bbox').is_not_null())
                .select([pl.col("x1_bbox").alias("x1"),
                         pl.col("y1_bbox").alias("y1"),
                         pl.col("x2_bbox").alias("x2"),
                         pl.col("y2_bbox").alias("y2")
                         ])
                .sort(['x1', 'y1', 'x2', 'y2'])
                .with_row_count(name="index")
                )

    return df_cells
