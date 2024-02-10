# coding: utf-8
import itertools

import polars as pl


def deduplicate_cells(df_cells: pl.LazyFrame) -> pl.LazyFrame:
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

    if df_cells.collect().height == 0:
        return df_cells

    # Cross join to get cells pairs and filter on right cells bigger than right cells
    df_cross_cells = (df_cells.clone()
                      .join(df_cells_cp, how='cross')
                      .filter(pl.col('index') != pl.col('index_'))
                      .filter(pl.col('area') <= pl.col('area_'))
                      )

    ### Compute indicator if the first cell is contained in second cell
    # Compute coordinates of intersection
    df_cross_cells = df_cross_cells.with_columns([pl.max_horizontal(['x1', 'x1_']).alias('x_left'),
                                                  pl.max_horizontal(['y1', 'y1_']).alias('y_top'),
                                                  pl.min_horizontal(['x2', 'x2_']).alias('x_right'),
                                                  pl.min_horizontal(['y2', 'y2_']).alias('y_bottom'),
                                                  ])

    # Compute area of intersection
    df_cross_cells = df_cross_cells.with_columns((pl.max_horizontal([pl.col('x_right') - pl.col('x_left'), pl.lit(0)])
                                                  * pl.max_horizontal([pl.col('y_bottom') - pl.col('y_top'), pl.lit(0)])
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
                      .with_columns(pl.min_horizontal([(pl.col(_1) - pl.col(_2)).abs()
                                                       for _1, _2 in itertools.product(['x1', 'x2'], ['x1_', 'x2_'])]
                                                      ).alias('diff_x'))
                      .with_columns(pl.min_horizontal([(pl.col(_1) - pl.col(_2)).abs()
                                                       for _1, _2 in itertools.product(['y1', 'y2'], ['y1_', 'y2_'])]
                                                      ).alias('diff_y'))
                      )

    # Create column indicating if both cells are adjacent and  column indicating if the right cell is redundant with
    # the left cell
    condition_adjacent = (((pl.col("overlapping_y") > 5) & (pl.col("diff_x") == 0))
                          | ((pl.col("overlapping_x") > 5) & (pl.col("diff_y") == 0))
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
    df_final_cells = (df_cells.with_row_index(name="cnt")
                      .filter(~pl.col('cnt').is_in(redundant_cells))
                      .drop('cnt')
                      )

    return df_final_cells
