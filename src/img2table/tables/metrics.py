# coding: utf-8
from typing import Tuple, Optional, List

import cv2
import numpy as np
import polars as pl
from numba import njit, prange

from img2table.tables import threshold_dark_areas, find_components
from img2table.tables.objects.cell import Cell


@njit("List(int64)(int32[:,:],int32[:,:])", fastmath=True, cache=True, parallel=False)
def remove_dots(cc_labels: np.ndarray, stats: np.ndarray) -> List[int]:
    """
    Remove dots from connected components
    :param cc_labels: connected components' label array
    :param stats: connected components' stats array
    :return: list of non-dot connected components' indexes
    """
    cc_to_keep = list()

    for idx in prange(len(stats)):
        if idx == 0:
            continue

        x, y, w, h, area = stats[idx][:]

        # Check number of inner pixels
        inner_pixels = 0
        for row in prange(y, y + h):
            prev_position = -1
            for col in range(x, x + w):
                value = cc_labels[row][col]
                if value == idx:
                    if prev_position >= 0:
                        inner_pixels += col - prev_position - 1
                    prev_position = col

        for col in prange(x, x + w):
            prev_position = -1
            for row in range(y, y + h):
                value = cc_labels[row][col]
                if value == idx:
                    if prev_position >= 0:
                        inner_pixels += row - prev_position - 1
                    prev_position = row

        # Compute roundness
        roundness = 4 * area / (np.pi * max(h, w) ** 2)

        if not (inner_pixels / (2 * area) <= 0.1 and roundness >= 0.7):
            cc_to_keep.append(idx)

    return cc_to_keep


def remove_dotted_lines(thresh: np.ndarray, cc_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove dotted lines in image by identifying aligned connected components
    :param thresh: thresholded image
    :param cc_array: connected components' array
    :return: updated thresholded image and filtered connected components' array
    """
    # Create dataframe of connected components
    df_cc = (pl.DataFrame([{"idx": idx, "x1": cc[0], "y1": cc[1], "x2": cc[0] + cc[2], "y2": cc[1] + cc[3]}
                           for idx, cc in enumerate(cc_array)])
             .with_columns((pl.col("y2") - pl.col("y1")).alias("height"),
                           (pl.col("x2") - pl.col("x1")).alias("width"),
                           ((pl.col("x1") + pl.col("x2")) / 2).alias("x"),
                           ((pl.col("y1") + pl.col("y2")) / 2).alias("y")
                           )
             )

    # Horizontal case
    df_hor_areas = (df_cc.filter(pl.col("width") / pl.col("height") >= 2)
                    .sort("y", "x")
                    .with_columns((pl.col("y").shift(-1) - pl.col("y") > 2).cast(int).alias("cluster"))
                    .fill_null(0)
                    .with_columns(pl.col("cluster").cum_sum())
                    .group_by("cluster")
                    .agg(pl.col("idx"),
                         pl.col("width").sum().alias("width_sum"),
                         pl.col("x1").min().alias("x1_area"),
                         pl.col("y1").median().cast(int).alias("y1_area"),
                         pl.col("x2").max().alias("x2_area"),
                         pl.col("y2").median().cast(int).alias("y2_area"))
                    .filter(pl.col("width_sum") / (pl.col("x2_area") - pl.col("x1_area")) >= 0.66,
                            pl.col("idx").list.len() >= 5)
                    .select("x1_area", "y1_area", "x2_area", "y2_area")
                    )

    # Vertical case
    df_ver_areas = (df_cc.filter(pl.col("height") / pl.col("width") >= 2)
                    .sort("x", "y")
                    .with_columns((pl.col("x").shift(-1) - pl.col("x") > 2).cast(int).alias("cluster"))
                    .fill_null(0)
                    .with_columns(pl.col("cluster").cum_sum())
                    .group_by("cluster")
                    .agg(pl.col("idx"),
                         pl.col("height").sum().alias("height_sum"),
                         pl.col("x1").median().cast(int).alias("x1_area"),
                         pl.col("y1").min().alias("y1_area"),
                         pl.col("x2").median().cast(int).alias("x2_area"),
                         pl.col("y2").max().alias("y2_area"))
                    .filter(pl.col("height_sum") / (pl.col("y2_area") - pl.col("y1_area")) >= 0.66,
                            pl.col("idx").list.len() >= 5)
                    .select("x1_area", "y1_area", "x2_area", "y2_area")
                    )

    # Identify cc to delete, i.e intersecting with found areas
    df_areas_to_delete = pl.concat([df_hor_areas, df_ver_areas])
    if df_areas_to_delete.height > 0:
        df_cc_to_delete = (df_cc.join(df_areas_to_delete, how="cross")
                           .with_columns(
            (pl.min_horizontal("x2", "x2_area") - pl.max_horizontal("x1", "x1_area")).alias("x_overlap"),
            (pl.min_horizontal("y2", "y2_area") - pl.max_horizontal("y1", "y1_area")).alias("y_overlap"),
            ((pl.col("x2") - pl.col("x1")) * (pl.col("y2") - pl.col("y1"))).alias("area"))
                           .filter(pl.col("x_overlap") > 0, pl.col("y_overlap") > 0)
                           .with_columns((pl.col("x_overlap") * pl.col("y_overlap")).alias("int_area"))
                           .group_by("idx", "area").agg(pl.col('int_area').sum())
                           .filter(pl.col("int_area") / pl.col('area') >= 0.25)
                           .select("idx")
                           .unique()
                           )

        cc_to_delete = [row.get('idx') for row in df_cc_to_delete.to_dicts()]

        # Remove dotted areas from thresholded image
        for row in df_areas_to_delete.to_dicts():
            thresh[row.get("y1_area"):row.get("y2_area"), row.get("x1_area"):row.get("x2_area")] = 0

        return thresh, np.delete(cc_array, cc_to_delete, axis=0)
    else:
        return thresh, cc_array


def update_character_detection(thresh: np.ndarray, stats: np.ndarray, char_length: float) -> List[Cell]:
    """
    Update character detection
    :param thresh: thresholded image
    :param stats: detection character connected components
    :param char_length: average character length
    :return: list of updated character cells
    """
    # Compute character cells
    character_cells = [Cell(x1=c[cv2.CC_STAT_LEFT],
                            y1=c[cv2.CC_STAT_TOP],
                            x2=c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH],
                            y2=c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT]) for c in stats]

    # Remove character cells for binary image
    for c in character_cells:
        thresh[c.y1:c.y2, c.x1:c.x2] = 0

    # Detect connected components
    _, cc_labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # Create dataframe from connected components and character cells
    df_new_cc = pl.LazyFrame([{"x1": c[cv2.CC_STAT_LEFT],
                               "y1": c[cv2.CC_STAT_TOP],
                               "x2": c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH],
                               "y2": c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT]} for c in stats])
    df_characters = pl.LazyFrame([{"x1_c": c.x1, "y1_c": c.y1, "x2_c": c.x2, "y2_c": c.y2}
                                  for c in character_cells])

    # Identify relevant connected components that correspond to characters
    df_relevant_cc = (df_new_cc.join(df_characters, how="cross")
                      # Compute statistics
                      .with_columns(height=pl.col('y2') - pl.col('y1'),
                                    height_c=pl.col('y2_c') - pl.col('y1_c'),
                                    width=pl.col('x2') - pl.col('x1'),
                                    width_c=pl.col('x2_c') - pl.col('x1_c'),
                                    x_overlap=pl.min_horizontal("x2", "x2_c") - pl.max_horizontal("x1", "x1_c"),
                                    y_overlap=pl.min_horizontal("y2", "y2_c") - pl.max_horizontal("y1", "y1_c"))
                      .with_columns(median_w=pl.col("width_c").median(),
                                    median_h=pl.col("height_c").median())
                      .filter(pl.col("width") / pl.col("height") <= 3)
                      # Filter cc that are aligned with existing characters
                      .filter((pl.col("x_overlap") >= 0.5 * pl.min_horizontal("width", "width_c"))
                              | (pl.col("y_overlap") >= 0.5 * pl.min_horizontal("height", "height_c"))
                              )
                      .filter(pl.max_horizontal("height", "width") <= 3 * pl.max_horizontal("height_c", "width_c"))
                      .with_columns(pl.min_horizontal((pl.col("x1") - pl.col("x1_c")).abs(),
                                                      (pl.col("x1") - pl.col("x2_c")).abs(),
                                                      (pl.col("x2") - pl.col("x1_c")).abs(),
                                                      (pl.col("x2") - pl.col("x2_c")).abs(),
                                                      ).alias("x_distance"),
                                    pl.min_horizontal((pl.col("y1") - pl.col("y1_c")).abs(),
                                                      (pl.col("y1") - pl.col("y2_c")).abs(),
                                                      (pl.col("y2") - pl.col("y1_c")).abs(),
                                                      (pl.col("y2") - pl.col("y2_c")).abs(),
                                                      ).alias("y_distance")
                                    )
                      .filter(((pl.col("y_overlap") > 0) & (pl.col("x_distance") <= char_length))
                              | ((pl.col("x_overlap") > 0) & (pl.col("y_distance") <= char_length)))
                      .select("x1", "y1", "x2", "y2")
                      .unique()
                      )

    # Create new characters cells
    new_characters_cells = [Cell(**c) for c in df_relevant_cc.collect().to_dicts()]

    return character_cells + new_characters_cells


def compute_char_length(img: np.ndarray) -> Tuple[Optional[float], Optional[List[Cell]]]:
    """
    Compute average character length based on connected components' analysis
    :param img: image array
    :return: tuple with average character length and character cells
    """
    # Thresholding
    thresh = threshold_dark_areas(img=img, char_length=11)

    # Connected components
    _, cc_labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # Remove dots
    cc_to_keep = remove_dots(cc_labels=cc_labels, stats=stats)
    stats = stats[cc_to_keep, :]

    # Remove connected components with less than 10 pixels
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 10
    stats = stats[mask_pixels]

    if len(stats) == 0:
        return None, None

    # Remove dotted lines
    thresh, stats = remove_dotted_lines(thresh=thresh, cc_array=stats)

    # Filter components based on aspect ratio
    mask_ar = (np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
               / np.minimum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])) <= 5

    # Filter components based on fill ratio
    mask_fill = stats[:, cv2.CC_STAT_AREA] / (stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]) > 0.08
    stats = stats[mask_ar & mask_fill]

    if len(stats) == 0:
        return None, None

    # Compute median width and height
    median_width = np.median(stats[:, cv2.CC_STAT_WIDTH])
    median_height = np.median(stats[:, cv2.CC_STAT_HEIGHT])

    # Compute bbox area bounds
    upper_bound = 5 * median_width * median_height
    lower_bound = 0.2 * median_width * median_height

    # Filter connected components according to their area
    mask_lower_area = lower_bound <= stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_area = upper_bound >= stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_area = mask_lower_area & mask_upper_area

    # Identify dashes
    mask_ar = stats[:, cv2.CC_STAT_WIDTH] / stats[:, cv2.CC_STAT_HEIGHT] >= 2
    mask_width_upper = stats[:, cv2.CC_STAT_WIDTH] <= 1.5 * median_width
    mask_width_lower = stats[:, cv2.CC_STAT_WIDTH] >= 0.5 * median_width
    mask_dash = mask_ar & mask_width_upper & mask_width_lower

    # Filter connected components from mask
    stats = stats[mask_area | mask_dash]

    if len(stats) > 0:
        # Compute average character length
        argmax_char_length = float(np.argmax(np.bincount(stats[:, cv2.CC_STAT_WIDTH])))
        mean_char_length = np.mean(stats[:, cv2.CC_STAT_WIDTH])
        char_length = mean_char_length if 1.5 * argmax_char_length <= mean_char_length else argmax_char_length

        # Update character detection
        updated_characters = update_character_detection(thresh=thresh,
                                                        stats=stats,
                                                        char_length=char_length)

        return char_length, updated_characters
    else:
        return None, None


def compute_merged_characters(character_cells: List[Cell], char_length: float) -> List[Cell]:
    """
    Identify characters that belong to the same word and create merged contours
    :param character_cells: list of character cells
    :param char_length: average character length
    :return: list of contour cells
    """
    # Create characters dataframe
    df_characters = (pl.DataFrame([{"idx": idx, "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2}
                                   for idx, c in enumerate(character_cells)])
                     .with_columns(height=pl.col('y2') - pl.col("y1"),
                                   width=pl.col("x2") - pl.col("x1"))
                     )

    # Cross join characters and compute pairwise metrics
    df_cross_chars = (df_characters.join(df_characters, how="cross")
                      .filter(pl.col('idx') != pl.col("idx_right"))
                      .with_columns(x_overlap=pl.min_horizontal("x2", "x2_right") - pl.max_horizontal("x1", "x1_right"),
                                    y_overlap=pl.min_horizontal("y2", "y2_right") - pl.max_horizontal("y1", "y1_right"),
                                    x_distance=pl.min_horizontal((pl.col("x1") - pl.col("x2_right")).abs(),
                                                                 (pl.col("x2") - pl.col("x1_right")).abs()),
                                    y_distance=pl.min_horizontal((pl.col("y1") - pl.col("y2_right")).abs(),
                                                                 (pl.col("y2") - pl.col("y1_right")).abs()),
                                    )
                      )

    # Identify characters that correspond horizontally
    df_horizontal_cc = (df_cross_chars
                        .filter((pl.col("x_distance") <= 0.5 * char_length)
                                & (pl.col("y_overlap") >= 0.5 * pl.min_horizontal("height", "height_right")))
                        .select("idx", "idx_right")
                        )

    # Identify characters that correspond vertically
    df_vertical_cc = (df_cross_chars
                      .join(df_horizontal_cc.select("idx"), on=["idx"], how="anti")
                      .join(df_horizontal_cc.select("idx_right"), on=["idx_right"], how="anti")
                      .filter((pl.col("y_distance") <= 0.5 * char_length)
                              & (pl.col("x_overlap") >= 0.5 * pl.min_horizontal("width", "width_right")))
                      .select("idx", "idx_right")
                      )

    # Identify remaining characters
    df_remaining_chars = (df_characters
                          .join(df_horizontal_cc.select("idx"), on=["idx"], how="anti")
                          .join(df_vertical_cc.select("idx"), on=["idx"], how="anti")
                          .filter(pl.max_horizontal("height", "width") >= 0.75 * char_length)
                          .select("idx")
                          )

    # Get list of adjacent character pairs
    adjacent_chars = [{row.get('idx'), row.get('idx_right')} for row in df_horizontal_cc.to_dicts()] + \
                     [{row.get('idx'), row.get('idx_right')} for row in df_vertical_cc.to_dicts()] + \
                     [{row.get('idx')} for row in df_remaining_chars.to_dicts()]

    # Create clusters of characters
    clusters = find_components(edges=adjacent_chars)

    # Create contours cells
    contours_cells = [Cell(x1=min([character_cells[idx].x1 for idx in cl]),
                           y1=min([character_cells[idx].y1 for idx in cl]),
                           x2=max([character_cells[idx].x2 for idx in cl]),
                           y2=max([character_cells[idx].y2 for idx in cl]))
                      for cl in clusters]

    return contours_cells


def compute_median_line_sep(img: np.ndarray, character_cells: List[Cell],
                            char_length: float) -> Tuple[Optional[float], Optional[List[Cell]]]:
    """
    Compute median separation between rows
    :param img: image array
    :param character_cells: list of character cells
    :param char_length: average character length
    :return: median separation between rows
    """
    # Identify characters that belong to the same word and create merged contours
    contours_cells = compute_merged_characters(character_cells=character_cells,
                                               char_length=char_length)

    if len(contours_cells) == 0:
        return None, []

    # Create contours dataframe
    df_contours = pl.LazyFrame(data=[{"id": idx, "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2}
                                     for idx, c in enumerate(contours_cells)])

    # Cross join to get corresponding contours and filter on contours that corresponds horizontally
    df_h_cnts = (df_contours.join(df_contours, how='cross')
                 .filter(pl.col('id') != pl.col('id_right'))
                 .filter(pl.min_horizontal(['x2', 'x2_right']) - pl.max_horizontal(['x1', 'x1_right']) > 0)
                 )

    # Get contour which is directly below
    df_cnts_below = (df_h_cnts.filter(pl.col('y1') < pl.col('y1_right'))
                     .sort(['id', 'y1_right'])
                     .with_columns(pl.lit(1).alias('ones'))
                     .with_columns(pl.col('ones').cum_sum().over(["id"]).alias('rk'))
                     .filter(pl.col('rk') == 1)
                     )

    if df_cnts_below.collect().height == 0:
        return None, contours_cells

    # Compute median vertical distance between contours
    median_v_dist = (df_cnts_below.with_columns(((pl.col('y1_right') + pl.col('y2_right')
                                                  - pl.col('y1') - pl.col('y2')) / 2).abs().alias('y_diff'))
                     .select(pl.median('y_diff'))
                     .collect()
                     .to_dicts()
                     .pop()
                     .get('y_diff')
                     )

    return median_v_dist, contours_cells


def compute_img_metrics(img: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[List[Cell]]]:
    """
    Compute metrics from image
    :param img: image array
    :return: average character length, median line separation and image contours
    """
    # Compute average character length based on connected components analysis
    char_length, character_cells = compute_char_length(img=img)

    if char_length is None:
        return None, None, None

    # Compute median separation between rows
    median_line_sep, contours = compute_median_line_sep(img=img,
                                                        character_cells=character_cells,
                                                        char_length=char_length)

    return char_length, median_line_sep, contours
