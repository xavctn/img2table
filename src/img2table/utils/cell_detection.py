# coding: utf-8
from typing import List

import pandas as pd

from img2table.objects.tables import Cell, Line


def get_cells_dataframe(horizontal_lines: List[Line], vertical_lines: List[Line]) -> pd.DataFrame:
    # Create dataframe from horizontal and vertical lines
    h_lines_values = [{"x1": line.x1, "x2": line.x2, "y1": line.y1,
                       "y2": line.y2, "width": line.width, "height": line.height}
                      for line in horizontal_lines]
    df_h_lines = pd.DataFrame(h_lines_values)

    v_lines_values = [{"x1": line.x1, "x2": line.x2, "y1": line.y1,
                       "y2": line.y2, "width": line.width, "height": line.height}
                      for line in vertical_lines]
    df_v_lines = pd.DataFrame(v_lines_values)

    # Create copy of df_h_lines
    df_h_lines_cp = df_h_lines.copy()
    df_h_lines_cp.columns = ["x1_", "x2_", "y1_", "y2_", 'width_', "height_"]

    # Cross join with itself to get pairs of horizontal lines
    cross_h_lines = df_h_lines.merge(df_h_lines_cp, how='cross')
    cross_h_lines = cross_h_lines[cross_h_lines["y1"] < cross_h_lines["y1_"]]

    # Compute horizontal correspondences between lines
    cross_h_lines["l_corresponds"] = (cross_h_lines["x1"] - cross_h_lines["x1_"] / cross_h_lines["width"]).abs() <= 0.02
    cross_h_lines["r_corresponds"] = (cross_h_lines["x2"] - cross_h_lines["x2_"] / cross_h_lines["width"]).abs() <= 0.02
    cross_h_lines["l_contained"] = (((cross_h_lines["x1"] <= cross_h_lines["x1_"])
                                    & (cross_h_lines["x1_"] <= cross_h_lines["x2"]))
                                    | ((cross_h_lines["x1_"] <= cross_h_lines["x1"])
                                       & (cross_h_lines["x1"] <= cross_h_lines["x2_"])))
    cross_h_lines["r_contained"] = (((cross_h_lines["x1"] <= cross_h_lines["x2_"])
                                     & (cross_h_lines["x2_"] <= cross_h_lines["x2"]))
                                    | ((cross_h_lines["x1_"] <= cross_h_lines["x2"])
                                       & (cross_h_lines["x2"] <= cross_h_lines["x2_"])))

    # Create condition on horizontal correspondence in order to use both lines and filter on relevant combinations
    matching_condition = ((cross_h_lines["l_corresponds"] | cross_h_lines["l_contained"])
                          & (cross_h_lines["r_corresponds"] | cross_h_lines["r_contained"]))
    cross_h_lines = cross_h_lines[matching_condition]

    # Create cell bbox from horizontal lines
    cross_h_lines["x1_bbox"] = cross_h_lines[["x1", "x1_"]].max(axis=1)
    cross_h_lines["x2_bbox"] = cross_h_lines[["x2", "x2_"]].min(axis=1)
    cross_h_lines["y1_bbox"] = cross_h_lines["y1"]
    cross_h_lines["y2_bbox"] = cross_h_lines["y1_"]
    df_bbox = cross_h_lines[["x1_bbox", "y1_bbox", "x2_bbox", "y2_bbox"]].reset_index()

    # Cross join with vertical lines
    df_bbox["h_margin"] = pd.concat([(df_bbox["x2_bbox"] - df_bbox["x1_bbox"]) * 0.05,
                                     pd.Series(5.0, index=range(len(df_bbox)))],
                                    axis=1).max(axis=1).round()
    df_bbox_v = df_bbox.merge(df_v_lines, how='cross')

    # Check horizontal correspondence between cell and vertical lines
    horizontal_cond = ((df_bbox_v["x1_bbox"] - df_bbox_v["h_margin"] <= df_bbox_v["x1"])
                       & (df_bbox_v["x2_bbox"] + df_bbox_v["h_margin"] > + df_bbox_v["x1"]))
    df_bbox_v = df_bbox_v[horizontal_cond]

    # Check vertical overlapping
    df_bbox_v["overlapping"] = df_bbox_v[["y2", "y2_bbox"]].min(axis=1) - df_bbox_v[["y1", "y1_bbox"]].max(axis=1)
    df_bbox_v = df_bbox_v[df_bbox_v["overlapping"] / (df_bbox_v["y2_bbox"] - df_bbox_v["y1_bbox"]) >= 0.8]

    # Get all vertical delimiters by bbox
    df_bbox_delimiters = (df_bbox_v.groupby(['index', "x1_bbox", "x2_bbox", "y1_bbox", "y2_bbox"])
                          .agg(dels=('x1', lambda x: [bound for bound in zip(sorted(x), sorted(x)[1:])] or None))
                          )

    # Create new cells based on vertical delimiters
    df_bbox_delimiters = df_bbox_delimiters[df_bbox_delimiters["dels"].notnull()].explode(column="dels").reset_index()
    df_bbox_delimiters[['del1', 'del2']] = pd.DataFrame(df_bbox_delimiters.dels.tolist(),
                                                        index=df_bbox_delimiters.index)
    df_bbox_delimiters["x1_bbox"] = df_bbox_delimiters["del1"]
    df_bbox_delimiters["x2_bbox"] = df_bbox_delimiters["del2"]

    # Reformat output dataframe
    df_cells = df_bbox_delimiters[["x1_bbox", "y1_bbox", "x2_bbox", "y2_bbox"]]
    df_cells.columns = ["x1", "y1", "x2", "y2"]

    return df_cells.reset_index()


def deduplicate_cells(df_cells: pd.DataFrame) -> pd.DataFrame:
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

    # Compute area of word bbox and intersection
    df_cross_cells["w_area"] = df_cross_cells["width"] * df_cross_cells["height"]
    df_cross_cells["int_area"] = (df_cross_cells["x_right"] - df_cross_cells["x_left"]) * (df_cross_cells["y_bottom"] - df_cross_cells["y_top"])

    # Create column indicating if left cell is contained in right cell
    df_cross_cells["contained"] = ((df_cross_cells["x_right"] >= df_cross_cells["x_left"])
                                   & (df_cross_cells["y_bottom"] >= df_cross_cells["y_top"])
                                   & (df_cross_cells["int_area"] / df_cross_cells["w_area"] >= 0.9))

    ### Compute indicator if cells are adjacent
    # Compute intersections and horizontal / vertical differences
    df_cross_cells["overlapping_x"] = df_cross_cells[["x2", "x2_"]].min(axis=1) - df_cross_cells[["x1", "x1_"]].max(axis=1)
    df_cross_cells["overlapping_y"] = df_cross_cells[["y2", "y2_"]].min(axis=1) - df_cross_cells[["y1", "y1_"]].max(axis=1)
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


def get_cells_v2(h_lines, v_lines):
    df_cells = get_cells_dataframe(h_lines, v_lines)
    dedup_cells = deduplicate_cells(df_cells)
    return dedup_cells


def get_cells(horizontal_lines: List[Line], vertical_lines: List[Line]) -> List[Cell]:
    """
    Identify cells from horizontal and vertical lines
    :param horizontal_lines: list of horizontal lines
    :param vertical_lines: list of vertical lines
    :return: list of all cells in image
    """
    # Create dataframe with cells from horizontal and vertical lines
    df_cells = get_cells_dataframe(horizontal_lines=horizontal_lines, vertical_lines=vertical_lines)

    # Deduplicate cells
    df_cells_dedup = deduplicate_cells(df_cells=df_cells)

    # Convert to Cell objects
    cells = [Cell(x1=row["x1"], x2=row["x2"], y1=row["y1"], y2=row["y2"])
             for row in df_cells_dedup.to_dict(orient='records')]

    return cells
