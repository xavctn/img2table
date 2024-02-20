# coding: utf-8
from itertools import groupby
from operator import itemgetter
from typing import List, Optional

import cv2
import numpy as np
import polars as pl

from img2table.tables import find_components
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line


def filter_dilation(thresh: np.ndarray, dilation: np.ndarray) -> np.ndarray:
    """
    Filter dilated image only on relevant areas (i.e where it links dotted lines)
    :param thresh: original threshold image
    :param dilation: dilated image
    :return: filtered dilated image
    """
    # Initialize filtered dilated image
    filtered_dilation = np.full(shape=thresh.shape, fill_value=0).astype(np.uint8)

    # Get connected components of dilated image
    _, _, cc_stats_dilated, _ = cv2.connectedComponentsWithStats(dilation, 8, cv2.CV_32S)

    for idx, stat in enumerate(cc_stats_dilated):
        if idx == 0:
            continue

        # Get stats
        x, y, w, h, area = stat

        # Get cropped original image and identify white pixels
        cropped = thresh[y:y + h, x:x + w]
        y_white_pixels, x_white_pixels = np.where(cropped == 255)

        # Compute relevant coordinates for dilation
        x_min, x_max = x + np.min(x_white_pixels), x + np.max(x_white_pixels)
        y_min, y_max = y + np.min(y_white_pixels), y + np.max(y_white_pixels)
        filtered_dilation[y_min:y_max, x_min:x_max] = dilation[y_min:y_max, x_min:x_max]

    return filtered_dilation


def dilate_dotted_lines(thresh: np.ndarray, char_length: float) -> np.ndarray:
    """
    Dilate specific rows/columns of the threshold image in order to detect dotted rows
    :param thresh: threshold image array
    :param char_length: average character length in image
    :return: threshold image with dilated dotted rows
    """
    # Compute non-null thresh and its average value
    non_null_thresh = thresh[:, np.max(thresh, axis=0) > 0]
    non_null_thresh = non_null_thresh[np.max(non_null_thresh, axis=1) > 0, :]
    w_mean = np.mean(non_null_thresh)

    ### Horizontal case
    # Create dilated image
    h_dilated = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(char_length), 1), 1)))

    # Get rows with at least 2 times the average number of white pixels
    h_non_null = np.where(np.max(thresh, axis=1) > 0)[0]
    white_rows = np.where(np.mean(thresh[:, min(h_non_null):max(h_non_null)], axis=1) > 4 * w_mean)[0].tolist()

    # Split into consecutive groups of rows and keep only small ones to avoid targeting text rows
    white_rows_cl = [list(map(itemgetter(1), g))
                     for k, g in groupby(enumerate(white_rows), lambda i_x: i_x[0] - i_x[1])]
    white_rows_final = [idx for cl in white_rows_cl for idx in cl if len(cl) < char_length]

    # Keep only dilated image on specific rows and filter it
    mask = np.ones(thresh.shape[0], dtype=bool)
    mask[white_rows_final] = False
    h_dilated[mask, :] = 0
    filtered_h_dilated = filter_dilation(thresh=thresh, dilation=h_dilated)

    ### Vertical case
    # Create dilated image
    v_dilated = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(char_length), 1))))

    # Get columns with at least 2 times the average number of white pixels
    v_non_null = np.where(np.max(thresh, axis=0) > 0)[0]
    white_cols = np.where(np.mean(thresh[min(v_non_null):max(v_non_null), :], axis=0) >= 4 * w_mean)[0].tolist()

    # Split into consecutive groups of cols and keep only small ones to avoid targeting text columns
    white_cols_cl = [list(map(itemgetter(1), g))
                     for k, g in groupby(enumerate(white_cols), lambda i_x: i_x[0] - i_x[1])]
    white_cols_final = [idx for cl in white_cols_cl for idx in cl if len(cl) < char_length]

    # Keep only dilated image on specific columns and filter it
    mask = np.ones(thresh.shape[1], dtype=bool)
    mask[white_cols_final] = False
    v_dilated[:, mask] = 0
    filtered_v_dilated = filter_dilation(thresh=thresh, dilation=v_dilated)

    # Update thresh
    new_thresh = np.maximum(thresh, filtered_h_dilated)
    new_thresh = np.maximum(new_thresh, filtered_v_dilated)

    return new_thresh


def line_from_cluster(line_cluster: List[Line]) -> Line:
    """
    Create single line from a cluster of consecutive lines
    :param line_cluster: cluster of consecutive lines
    :return: resulting line from cluster
    """
    if all([line.vertical for line in line_cluster]):
        # Vertical case
        x1 = int(min([line.x1 - line.thickness // 2 for line in line_cluster]))
        x2 = int(max([line.x2 + line.thickness // 2 for line in line_cluster]))
        y1 = int(min([line.y1 for line in line_cluster]))
        y2 = int(max([line.y2 for line in line_cluster]))
        thickness = x2 - x1 + 1
        return Line(x1=(x1 + x2) // 2, y1=y1, x2=(x1 + x2) // 2, y2=y2, thickness=thickness)
    else:
        # Horizontal case
        y1 = int(min([line.y1 - line.thickness // 2 for line in line_cluster]))
        y2 = int(max([line.y2 + line.thickness // 2 for line in line_cluster]))
        x1 = int(min([line.x1 for line in line_cluster]))
        x2 = int(max([line.x2 for line in line_cluster]))
        thickness = y2 - y1 + 1
        return Line(x1=x1, y1=(y1 + y2) // 2, x2=x2, y2=(y1 + y2) // 2, thickness=thickness)


def merge_lines(lines: List[Line], max_gap: float) -> List[Line]:
    """
    Merge consecutive lines
    :param lines: list of detected lines
    :param max_gap: maximum gap between consecutive lines in order to merge them
    :return: list of merged lines
    """
    if len(lines) == 0:
        return []

    # Create lines dataframe
    df_lines = pl.DataFrame(data=[{**line.dict, **{"id_line": idx, "vertical": line.vertical}}
                                  for idx, line in enumerate(lines)])

    # Distance computations
    v_dist = ((pl.col("x1") - pl.col("x1_right")).pow(2) + (pl.col("y1") - pl.col("y2_right")).pow(2)).pow(0.5)
    h_dist = ((pl.col("y1") - pl.col("y1_right")).pow(2) + (pl.col("x1") - pl.col("x2_right")).pow(2)).pow(0.5)

    # Overlap computations
    v_overlap = pl.min_horizontal("y2", "y2_right") - pl.max_horizontal("y1", "y1_right")
    h_overlap = pl.min_horizontal("x2", "x2_right") - pl.max_horizontal("x1", "x1_right")

    # Cross join lines and identify consecutive lines
    df_cross_lines = (df_lines.join(df_lines, on=["vertical"], how='inner')
                      .with_columns(pl.when(pl.col('vertical')).then(v_dist).otherwise(h_dist).alias('distance'),
                                    pl.when(pl.col('vertical')).then(v_overlap).otherwise(h_overlap).alias('overlap'))
                      .filter(pl.col('distance') <= max_gap,
                              pl.col("overlap") <= max_gap)
                      .select("id_line", "id_line_right")
                      .unique()
                      )

    # Get list of consecutive lines
    consecutive_lines = [{idx} for idx in range(len(lines))]
    consecutive_lines += [{row.get('id_line'), row.get('id_line_right')} for row in df_cross_lines.to_dicts()]

    # Loop over couples to create clusters
    clusters = find_components(edges=consecutive_lines)

    # Get merged lines from clusters
    list_lines_clusters = [line_from_cluster(line_cluster=[lines[idx] for idx in cl]) for cl in clusters]

    return list_lines_clusters


def detect_lines(thresh: np.ndarray, contours: Optional[List[Cell]], char_length: Optional[float],
                 min_line_length: Optional[float]) -> (List[Line], List[Line]):
    """
    Detect horizontal and vertical rows on image
    :param thresh: thresholded image array
    :param contours: list of image contours as cell objects
    :param char_length: average character length
    :param min_line_length: minimum line length
    :return: horizontal and vertical rows
    """
    # Remove contours from thresh image
    for c in contours:
        thresh[c.y1:c.y2, c.x1:c.x2] = 0

    if char_length is not None:
        # Process threshold image in order to detect dotted rows
        thresh = dilate_dotted_lines(thresh=thresh, char_length=char_length)

    # Identify both vertical and horizontal rows
    for kernel_dims in [(min_line_length, 1), (1, min_line_length)]:
        # Apply masking on image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
        mask = cv2.morphologyEx(thresh.copy(), cv2.MORPH_OPEN, kernel, iterations=1)

        # Get stats
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

        lines = list()
        # Get relevant CC that correspond to lines
        for idx, stat in enumerate(stats):
            if idx == 0:
                continue

            # Get stats
            x, y, w, h, area = stat

            # Filter on aspect ratio
            if max(w, h) / min(w, h) < 5 and min(w, h) >= char_length:
                continue

            if w >= h:
                line = Line(x1=x, y1=y + h // 2, x2=x + w, y2=y + h // 2, thickness=h)
            else:
                line = Line(x1=x + w // 2, y1=y, x2=x + w // 2, y2=y + h, thickness=w)
            lines.append(line)

        # Merge lines
        merged_lines = merge_lines(lines=lines, max_gap=min_line_length / 3)

        yield merged_lines
