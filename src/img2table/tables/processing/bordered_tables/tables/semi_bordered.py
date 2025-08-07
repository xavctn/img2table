
import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.bordered_tables.tables.table_creation import normalize_table_cells


def get_lines_in_cluster(cluster: list[Cell], lines: list[Line]) -> tuple[list[Line], list[Line]]:
    """
    Identify list of lines belonging to cluster
    :param cluster: list of cells in cluster
    :param lines: list of lines in image
    :return: list of horizontal and vertical lines in cluster
    """
    # Compute cluster coordinates
    x_min, x_max = min([c.x1 for c in cluster]), max([c.x2 for c in cluster])
    y_min, y_max = min([c.y1 for c in cluster]), max([c.y2 for c in cluster])

    # Find horizontal and vertical lines of the cluster
    y_values_cl = {c.y1 for c in cluster}.union({c.y2 for c in cluster})
    h_lines_cl = [line for line in lines if line.horizontal
                  and min([abs(line.y1 - y) for y in y_values_cl]) <= 0.05 * (y_max - y_min)]

    # Find vertical lines of the cluster
    x_values_cl = {c.x1 for c in cluster}.union({c.x2 for c in cluster})
    v_lines_cl = [line for line in lines if line.vertical
                  and min([abs(line.x1 - x) for x in x_values_cl]) <= 0.05 * (x_max - x_min)]

    return h_lines_cl, v_lines_cl


def identify_table_dimensions(cluster: list[Cell], h_lines_cl: list[Line], v_lines_cl: list[Line],
                              char_length: float) -> tuple[int, int, int, int]:
    """
    Identify table dimensions by checking lines corresponding to cluster
    :param cluster: cluster of cells
    :param h_lines_cl: list of horizontal lines in cluster
    :param v_lines_cl: list of vertical lines in cluster
    :param char_length: average character length
    :return: tuple of cluster dimensions
    """
    if h_lines_cl:
        # Compute extrema dimensions of lines
        left, right = min([line.x1 for line in h_lines_cl]), max([line.x2 for line in h_lines_cl])

        # Left end
        left_end_lines = [line for line in h_lines_cl if line.x1 - left <= 0.05 * (right - left)]
        if len({h_lines_cl[0], h_lines_cl[-1]}.difference(set(left_end_lines))) == 0:
            left_val = min([c.x1 for c in cluster]) if min([c.x1 for c in cluster]) - left <= 2 * char_length else left
        else:
            left_val = min([c.x1 for c in cluster])

        # Right end
        right_end_lines = [line for line in h_lines_cl if right - line.x2 <= 0.05 * (right - left)]
        if len({h_lines_cl[0], h_lines_cl[-1]}.difference(set(right_end_lines))) == 0:
            right_val = max([c.x2 for c in cluster]) if right - max([c.x2 for c in cluster]) <= 2 * char_length else right
        else:
            right_val = max([c.x2 for c in cluster])
    else:
        left_val, right_val = min([c.x1 for c in cluster]), max([c.x2 for c in cluster])

    if v_lines_cl:
        # Compute extrema dimensions of lines
        top, bottom = min([line.y1 for line in v_lines_cl]), max([line.y2 for line in v_lines_cl])

        # Top end
        top_end_lines = [line for line in v_lines_cl if line.y1 - top <= 0.05 * (bottom - top)]
        if len({v_lines_cl[0], v_lines_cl[-1]}.difference(set(top_end_lines))) == 0:
            top_val = min([c.y1 for c in cluster]) if min([c.y1 for c in cluster]) - top <= 2 * char_length else top
        else:
            top_val = min([c.y1 for c in cluster])

        # Bottom end
        bottom_end_lines = [line for line in v_lines_cl if bottom - line.y2 <= 0.05 * (bottom - top)]
        if len({v_lines_cl[0], v_lines_cl[-1]}.difference(set(bottom_end_lines))) == 0:
            bottom_val = max([c.y2 for c in cluster]) if bottom - max([c.y2 for c in cluster]) <= 2 * char_length else bottom
        else:
            bottom_val = max([c.y2 for c in cluster])
    else:
        top_val, bottom_val = min([c.y1 for c in cluster]), max([c.y2 for c in cluster])

    return left_val, right_val, top_val, bottom_val


def identify_potential_new_cells(cluster: list[Cell], h_lines_cl: list[Line], v_lines_cl: list[Line], left_val: int,
                                 right_val: int, top_val: int, bottom_val: int) -> list[Cell]:
    """
    Indetify potential new cells in cluster based on computed dimensions
    :param cluster: cluster of cells
    :param h_lines_cl: list of horizontal lines in cluster
    :param v_lines_cl: list of vertical lines in cluster
    :param left_val: left coordinate of cluster
    :param right_val: right coordinate of cluster
    :param top_val: top coordinate of cluster
    :param bottom_val: bottom coordinate of cluster
    :return: list of potential cells
    """
    # Compute x and y values used in cluster
    x_cluster = sorted({c.x1 for c in cluster}.union({c.x2 for c in cluster}).union({left_val, right_val}))
    y_cluster = sorted({c.y1 for c in cluster}.union({c.y2 for c in cluster}).union({top_val, bottom_val}))

    # Create list of new cells
    new_cells = []

    # Compute cells on left end
    x1, x2 = x_cluster[0], x_cluster[1]
    y_vals = sorted({top_val, bottom_val}.union({ln.y1 for ln in h_lines_cl if min(ln.x2, x2) - max(ln.x1, x1) >= 0.9 * (x2 - x1)}))
    for y1, y2 in zip(y_vals, y_vals[1:]):
        new_cell = Cell(x1=x1, y1=y1, x2=x2, y2=y2)
        if new_cell.area > 0:
            new_cells.append(new_cell)

    # Compute cells on right end
    x1, x2 = x_cluster[-2], x_cluster[-1]
    y_vals = sorted({top_val, bottom_val}.union({ln.y1 for ln in h_lines_cl if min(ln.x2, x2) - max(ln.x1, x1) >= 0.9 * (x2 - x1)}))
    for y1, y2 in zip(y_vals, y_vals[1:]):
        new_cell = Cell(x1=x1, y1=y1, x2=x2, y2=y2)
        if new_cell.area > 0:
            new_cells.append(new_cell)

    # Compute cells on top end
    y1, y2 = y_cluster[0], y_cluster[1]
    x_vals = sorted({left_val, right_val}.union({ln.x1 for ln in v_lines_cl if min(ln.y2, y2) - max(ln.y1, y1) >= 0.9 * (y2 - y1)}))
    for x1, x2 in zip(x_vals, x_vals[1:]):
        new_cell = Cell(x1=x1, y1=y1, x2=x2, y2=y2)
        if new_cell.area > 0:
            new_cells.append(new_cell)

    # Compute cells on bottom end
    y1, y2 = y_cluster[-2], y_cluster[-1]
    x_vals = sorted({left_val, right_val}.union({ln.x1 for ln in v_lines_cl if min(ln.y2, y2) - max(ln.y1, y1) >= 0.9 * (y2 - y1)}))
    for x1, x2 in zip(x_vals, x_vals[1:]):
        new_cell = Cell(x1=x1, y1=y1, x2=x2, y2=y2)
        if new_cell.area > 0:
            new_cells.append(new_cell)

    return list(set(new_cells))


def update_cluster_cells(cluster: list[Cell], new_cells: list[Cell]) -> list[Cell]:
    """
    Update cluster cells with new ones if relevant
    :param cluster: cluster of cells
    :param new_cells: list of potential new cells
    :return: list of updated cluster cells
    """
    if len(new_cells) == 0:
        return cluster

    # Create dataframe of cluster and new cells
    df_cluster = pl.DataFrame([{"x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2, "area": c.area} for c in cluster])
    df_cells = pl.DataFrame([{"idx": idx, "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2, "area": c.area}
                             for idx, c in enumerate(new_cells)])

    # Identify cells that do not overlap any other cell in the cluster
    df_cells_indep = (df_cells.join(df_cluster, how="cross")
                      .with_columns(x_overlap=pl.min_horizontal("x2", "x2_right") - pl.max_horizontal("x1", "x1_right"),
                                    y_overlap=pl.min_horizontal("y2", "y2_right") - pl.max_horizontal("y1", "y1_right"))
                      .with_columns(x_overlap=pl.max_horizontal(pl.lit(0), "x_overlap"),
                                    y_overlap=pl.max_horizontal(pl.lit(0), "y_overlap"))
                      .with_columns(area_overlap=pl.col("x_overlap") * pl.col("y_overlap"))
                      .with_columns(pct_overlap=pl.col("area_overlap") / pl.min_horizontal("area", "area_right"))
                      .with_columns(max_overlap=pl.col("pct_overlap").max().over("idx"))
                      .filter(pl.col("max_overlap") < 0.5)
                      .select("idx", "x1", "y1", "x2", "y2", "area")
                      .unique()
                      )

    if df_cells_indep.height == 0:
        return cluster

    # Remove cells that are duplicates
    df_dups = (df_cells_indep.join(df_cells_indep, how="cross")
               .filter(pl.col("area") <= pl.col("area_right"),
                       pl.col("idx") != pl.col("idx_right"))
               .with_columns(x_overlap=pl.min_horizontal("x2", "x2_right") - pl.max_horizontal("x1", "x1_right"),
                             y_overlap=pl.min_horizontal("y2", "y2_right") - pl.max_horizontal("y1", "y1_right"))
               .with_columns(x_overlap=pl.max_horizontal(pl.lit(0), "x_overlap"),
                             y_overlap=pl.max_horizontal(pl.lit(0), "y_overlap"))
               .with_columns(area_overlap=pl.col("x_overlap") * pl.col("y_overlap"))
               .with_columns(pct_overlap=pl.col("area_overlap") / pl.min_horizontal("area", "area_right"))
               .with_columns(max_overlap=pl.col("pct_overlap").max().over("idx"))
               .filter(pl.col("max_overlap") >= 0.5)
               .select("idx").unique())

    # Get list of final cells to be added
    df_final_cells = (df_cells_indep.join(df_dups, on=["idx"], how="anti")
                      .select("x1", "y1", "x2", "y2"))
    final_cells = [Cell(**row) for row in df_final_cells.to_dicts()]

    if final_cells:
        return normalize_table_cells(cluster_cells=cluster + final_cells)
    return cluster


def add_semi_bordered_cells(cluster: list[Cell], lines: list[Line], char_length: float) -> list[Cell]:
    """
    Identify and add semi-bordered cells to cluster
    :param cluster: cluster of cells
    :param lines: lines in image
    :param char_length: average character length
    :return: cluster with add semi-bordered cells
    """
    if len(cluster) == 0:
        return cluster

    # Identify lines belonging to cluster
    h_lines_cl, v_lines_cl = get_lines_in_cluster(cluster=cluster, lines=lines)

    # Identify table dimensions
    left_val, right_val, top_val, bottom_val = identify_table_dimensions(cluster=cluster,
                                                                         h_lines_cl=h_lines_cl,
                                                                         v_lines_cl=v_lines_cl,
                                                                         char_length=char_length)

    # Create potential new cells
    new_cells = identify_potential_new_cells(cluster=cluster,
                                             h_lines_cl=h_lines_cl,
                                             v_lines_cl=v_lines_cl,
                                             left_val=left_val,
                                             right_val=right_val,
                                             top_val=top_val,
                                             bottom_val=bottom_val)

    # Update cluster cells
    return update_cluster_cells(cluster=cluster, new_cells=new_cells)
