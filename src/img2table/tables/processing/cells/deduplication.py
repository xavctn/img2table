# coding: utf-8
import itertools

import polars as pl


def deduplicate_cells_vertically(df_cells: pl.LazyFrame) -> pl.LazyFrame:
    """
    Deduplicate cells that have a common upper or lower bound
    :param df_cells: dataframe containing cells
    :return: dataframe with deduplicate cells that have a common upper or lower bound
    """
    orig_cols = df_cells.columns

    # Deduplicate on upper bound
    df_cells = (df_cells.sort(by=["x1", "x2", "y1", "y2"])
                .with_columns(pl.lit(1).alias('ones'))
                .with_columns(pl.col('ones').cumsum().over(["x1", "x2", "y1"]).alias('cell_rk'))
                .filter(pl.col('cell_rk') == 1)
                )

    # Deduplicate on lower bound
    df_cells = (df_cells.sort(by=["x1", "x2", "y2", "y1"], reverse=[False, False, False, True])
                .with_columns(pl.lit(1).alias('ones'))
                .with_columns(pl.col('ones').cumsum().over(["x1", "x2", "y2"]).alias('cell_rk'))
                .filter(pl.col('cell_rk') == 1)
                )

    return df_cells.select(orig_cols)


def deduplicate_nested_cells(df_cells: pl.LazyFrame) -> pl.LazyFrame:
    """
    Deduplicate nested cells in order to keep the smallest ones
    :param df_cells: dataframe containing cells
    :return: dataframe containing cells after deduplication of the nested ones
    """
    # Create columns corresponding to cell characteristics
    df_cells = (df_cells.with_columns([(pl.col('x2') - pl.col('x1')).alias('width'),
                                       (pl.col('y2') - pl.col('y1')).alias('height')])
                .with_columns((pl.col('height') * pl.col('width')).alias('area'))
                )

    # Create copy of df_cells
    df_cells_cp = (df_cells.clone()
                   .rename({col: f"{col}_" for col in df_cells.columns})
                   )

    # Cross join to get cells pairs and filter on right cells bigger than right cells
    df_cross_cells = (df_cells.clone()
                      .join(df_cells_cp, how='cross')
                      .filter(pl.col('index') != pl.col('index_'))
                      .filter(pl.col('area') <= pl.col('area_'))
                      )

    ### Compute indicator if the first cell is contained in second cell
    # Compute coordinates of intersection
    df_cross_cells = df_cross_cells.with_columns([pl.max([pl.col('x1'), pl.col('x1_')]).alias('x_left'),
                                                  pl.max([pl.col('y1'), pl.col('y1_')]).alias('y_top'),
                                                  pl.min([pl.col('x2'), pl.col('x2_')]).alias('x_right'),
                                                  pl.min([pl.col('y2'), pl.col('y2_')]).alias('y_bottom'),
                                                  ])

    # Compute area of intersection
    df_cross_cells = df_cross_cells.with_columns((pl.max([pl.col('x_right') - pl.col('x_left'), pl.lit(0)])
                                                  * pl.max([pl.col('y_bottom') - pl.col('y_top'), pl.lit(0)])
                                                  ).alias('int_area')
                                                 )

    # Create column indicating if left cell is contained in right cell
    df_cross_cells = df_cross_cells.with_columns(((pl.col('x_right') >= pl.col('x_left'))
                                                  & (pl.col('y_bottom') >= pl.col('y_top'))
                                                  & (pl.col('int_area') / pl.col('area') >= 0.9)
                                                  ).alias('contained')
                                                 )

    ### Compute indicator if cells are adjacent
    # Compute intersections and horizontal / vertical differences
    df_cross_cells = (df_cross_cells
                      .with_columns([(pl.col('x_right') - pl.col('x_left')).alias('overlapping_x'),
                                     (pl.col('y_bottom') - pl.col('y_top')).alias('overlapping_y')])
                      .with_columns(pl.min([(pl.col(_1) - pl.col(_2)).abs()
                                            for _1, _2 in itertools.product(['x1', 'x2'], ['x1_', 'x2_'])]
                                           ).alias('diff_x'))
                      .with_columns(pl.min([(pl.col(_1) - pl.col(_2)).abs()
                                            for _1, _2 in itertools.product(['y1', 'y2'], ['y1_', 'y2_'])]
                                           ).alias('diff_y'))
                      )

    # Create column indicating if both cells are adjacent and  column indicating if the right cell is redundant with
    # the left cell
    condition_adjacent = (((pl.col("overlapping_y") > 5)
                           & (pl.col("diff_x") / pl.max([pl.col("width"), pl.col("width_")]) <= 0.05))
                          | ((pl.col("overlapping_x") > 5)
                             & (pl.col("diff_y") / pl.max([pl.col("height"), pl.col("height_")]) <= 0.05))
                          )
    df_cross_cells = (df_cross_cells.with_columns(condition_adjacent.alias('adjacent'))
                      .with_columns((pl.col('contained') & pl.col('adjacent')).alias('redundant'))
                      )

    # Get list of redundant cells and remove them from original cell dataframe
    redundant_cells = (df_cross_cells.filter(pl.col('redundant'))
                       .collect()
                       .get_column('index_')
                       .unique()
                       .to_list()
                       )
    df_final_cells = (df_cells.with_row_count(name="cnt")
                      .filter(~pl.col('cnt').is_in(redundant_cells))
                      .drop('cnt')
                      )

    return df_final_cells


def deduplicate_cells(df_cells: pl.LazyFrame) -> pl.LazyFrame:
    """
    Deduplicate cells dataframe
    :param df_cells: dataframe containing cells
    :return: dataframe with deduplicated cells
    """
    # Deduplicate cells by vertical positions
    df_cells = deduplicate_cells_vertically(df_cells=df_cells)

    # Deduplicate nested cells
    df_cells_final = deduplicate_nested_cells(df_cells=df_cells)

    return df_cells_final
