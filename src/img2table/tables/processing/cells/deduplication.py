# coding: utf-8

import pandas as pd


def deduplicate_cells_vertically(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate cells that have a common upper or lower bound
    :param df_cells: dataframe containing cells
    :return: dataframe with deduplicate cells that have a common upper or lower bound
    """
    # Get original columns
    orig_cols = df_cells.columns

    # Deduplicate on upper bound
    df_cells = df_cells.sort_values(by=["x1", "x2", "y1", "y2"])
    df_cells["cell_rk"] = df_cells.groupby(["x1", "x2", "y1"]).cumcount()
    df_cells = df_cells[df_cells["cell_rk"] == 0]

    # Deduplicate on lower bound
    df_cells = df_cells.sort_values(by=["x1", "x2", "y2", "y1"], ascending=[True, True, True, False])
    df_cells["cell_rk"] = df_cells.groupby(["x1", "x2", "y2"]).cumcount()
    df_cells = df_cells[df_cells["cell_rk"] == 0]

    return df_cells[orig_cols]


def deduplicate_nested_cells(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate nested cells in order to keep the smallest ones
    :param df_cells: dataframe containing cells
    :return: dataframe containing cells after deduplication of the nested ones
    """
    # Create columns corresponding to cell characteristics
    df_cells["width"] = df_cells["x2"] - df_cells["x1"]
    df_cells["height"] = df_cells["y2"] - df_cells["y1"]
    df_cells["area"] = df_cells["width"] * df_cells["height"]

    # Create copy of df_cells
    df_cells_cp = df_cells.copy()
    df_cells_cp.columns = ["index_", "x1_", "y1_", "x2_", "y2_", "width_", "height_", "area_"]

    # Cross join to get cells pairs and filter on right cells bigger than right cells
    df_cross_cells = df_cells.reset_index().merge(df_cells_cp, how='cross')
    df_cross_cells = df_cross_cells[df_cross_cells["index"] != df_cross_cells["index_"]]
    df_cross_cells = df_cross_cells[df_cross_cells["area"] <= df_cross_cells["area_"]]

    ### Compute indicator if the first cell is contained in second cell
    # Compute coordinates of intersection
    df_cross_cells["x_left"] = df_cross_cells[["x1", "x1_"]].max(axis=1)
    df_cross_cells["y_top"] = df_cross_cells[["y1", "y1_"]].max(axis=1)
    df_cross_cells["x_right"] = df_cross_cells[["x2", "x2_"]].min(axis=1)
    df_cross_cells["y_bottom"] = df_cross_cells[["y2", "y2_"]].min(axis=1)

    # Compute area of intersection
    df_cross_cells["int_area"] = (df_cross_cells["x_right"] - df_cross_cells["x_left"]) \
                                 * (df_cross_cells["y_bottom"] - df_cross_cells["y_top"])

    # Create column indicating if left cell is contained in right cell
    df_cross_cells["contained"] = ((df_cross_cells["x_right"] >= df_cross_cells["x_left"])
                                   & (df_cross_cells["y_bottom"] >= df_cross_cells["y_top"])
                                   & (df_cross_cells["int_area"] / df_cross_cells["area"] >= 0.9))

    ### Compute indicator if cells are adjacent
    # Compute intersections and horizontal / vertical differences
    df_cross_cells["overlapping_x"] = df_cross_cells["x_right"] - df_cross_cells["x_left"]
    df_cross_cells["overlapping_y"] = df_cross_cells["y_bottom"] - df_cross_cells["y_top"]
    df_cross_cells["diff_x"] = pd.concat([(df_cross_cells["x2"] - df_cross_cells["x1_"]).abs(),
                                          (df_cross_cells["x1"] - df_cross_cells["x2_"]).abs(),
                                          (df_cross_cells["x1"] - df_cross_cells["x1_"]).abs(),
                                          (df_cross_cells["x2"] - df_cross_cells["x2_"]).abs()],
                                         axis=1).min(axis=1)
    df_cross_cells["diff_y"] = pd.concat([(df_cross_cells["y1"] - df_cross_cells["y1_"]).abs(),
                                          (df_cross_cells["y2"] - df_cross_cells["y1_"]).abs(),
                                          (df_cross_cells["y1"] - df_cross_cells["y2_"]).abs(),
                                          (df_cross_cells["y2"] - df_cross_cells["y2_"]).abs()],
                                         axis=1).min(axis=1)

    # Create column indicating if both cells are adjacent
    condition_adjacent = (((df_cross_cells["overlapping_y"] > 5)
                           & (df_cross_cells["diff_x"] / df_cross_cells[["width", "width_"]].max(axis=1) <= 0.05))
                          | ((df_cross_cells["overlapping_x"] > 5)
                             & (df_cross_cells["diff_y"] / df_cross_cells[["height", "height_"]].max(axis=1) <= 0.05))
                          )
    df_cross_cells["adjacent"] = condition_adjacent

    # Create column indicating if the right cell is redundant with the left cell
    df_cross_cells["redundant"] = df_cross_cells["contained"] & df_cross_cells["adjacent"]

    # Get list of redundant cells and remove them from original cell dataframe
    redundant_cells = df_cross_cells[df_cross_cells["redundant"]]['index_'].drop_duplicates().values.tolist()
    df_final_cells = df_cells.drop(labels=redundant_cells)

    return df_final_cells


def deduplicate_cells(df_cells: pd.DataFrame) -> pd.DataFrame:
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
