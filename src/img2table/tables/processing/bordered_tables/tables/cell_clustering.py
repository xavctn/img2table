
import polars as pl

from img2table.tables import find_components
from img2table.tables.objects.cell import Cell


def get_adjacent_cells(cells: list[Cell]) -> list[set[int]]:
    """
    Identify adjacent cells
    :param cells: list of cells
    :return: list of sets of adjacent cells indexes
    """
    if len(cells) == 0:
        return []

    df_cells = pl.DataFrame([{"idx": idx, "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2, "height": c.height,
                              "width": c.width}
                             for idx, c in enumerate(cells)])

    # Crossjoin and identify adjacent cells
    df_adjacent_cells = (
        df_cells.join(df_cells, how='cross')
        # Compute horizontal and vertical overlap
        .with_columns((pl.min_horizontal(['x2', 'x2_right']) - pl.max_horizontal(['x1', 'x1_right'])).alias("x_overlap"),
                      (pl.min_horizontal(['y2', 'y2_right']) - pl.max_horizontal(['y1', 'y1_right'])).alias("y_overlap")
                      )
        # Compute horizontal and vertical differences
        .with_columns(
            pl.min_horizontal((pl.col('x1') - pl.col('x1_right')).abs(),
                              (pl.col('x1') - pl.col('x2_right')).abs(),
                              (pl.col('x2') - pl.col('x1_right')).abs(),
                              (pl.col('x2') - pl.col('x2_right')).abs()
                              ).alias('diff_x'),
            pl.min_horizontal((pl.col('y1') - pl.col('y1_right')).abs(),
                              (pl.col('y1') - pl.col('y2_right')).abs(),
                              (pl.col('y2') - pl.col('y1_right')).abs(),
                              (pl.col('y2') - pl.col('y2_right')).abs()
                              ).alias('diff_y')
        )
        # Compute thresholds for horizontal and vertical differences
        .with_columns(
            pl.min_horizontal(pl.lit(5), 0.05 * pl.min_horizontal(pl.col('width'), pl.col('width_right'))).alias('thresh_x'),
            pl.min_horizontal(pl.lit(5), 0.05 * pl.min_horizontal(pl.col('height'), pl.col('height_right'))).alias('thresh_y')
        )
        # Filter adjacent cells
        .filter(
           ((pl.col('y_overlap') > 5) & (pl.col('diff_x') <= pl.col('thresh_x')))
            | ((pl.col('x_overlap') > 5) & (pl.col('diff_y') <= pl.col('thresh_y')))
        )
        .select("idx", "idx_right")
        .unique()
        .sort(by=['idx', 'idx_right'])
    )

    # Get sets of adjacent cells indexes
    return [{row.get('idx'), row.get('idx_right')} for row in df_adjacent_cells.to_dicts()]


def cluster_cells_in_tables(cells: list[Cell]) -> list[list[Cell]]:
    """
    Based on adjacent cells, create clusters of cells that corresponds to tables
    :param cells: list cells in image
    :return: list of list of cells, representing several clusters of cells that form a table
    """
    # Get couples of adjacent cells
    adjacent_cells = get_adjacent_cells(cells=cells)

    # Loop over couples to create clusters
    clusters = find_components(edges=adjacent_cells)

    # Return list of cell objects
    return [[cells[idx] for idx in cl] for cl in clusters]
