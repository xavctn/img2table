# coding: utf-8
from typing import List

import pandas as pd

from img2table.tables.objects.line import Line


def get_cells_dataframe(horizontal_lines: List[Line], vertical_lines: List[Line]) -> pd.DataFrame:
    """
    Create dataframe of all possible cells from horizontal and vertical lines
    :param horizontal_lines: list of horizontal lines
    :param vertical_lines: list of vertical lines
    :return: dataframe containing all cells
    """
    default_df = pd.DataFrame(columns=["x1", "x2", "y1", "y2", 'width', "height"])
    # Create dataframe from horizontal and vertical lines
    df_h_lines = pd.DataFrame(map(lambda l: l.dict, horizontal_lines)) if horizontal_lines else default_df.copy()
    df_v_lines = pd.DataFrame(map(lambda l: l.dict, vertical_lines)) if vertical_lines else default_df.copy()

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

    try:
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
    except ValueError:
        return pd.DataFrame(columns=["index", "x1", "y1", "x2", "y2"])
