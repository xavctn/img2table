# coding: utf-8
from itertools import groupby
from operator import itemgetter
from typing import List, Optional, Dict

import cv2
import numpy as np
import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line


def threshold_dark_areas(img: np.ndarray, char_length: Optional[float]) -> np.ndarray:
    """
    Threshold image by differentiating areas with light and dark backgrounds
    :param img: image array
    :param char_length: average character length
    :return: threshold image
    """
    # Get threshold on image and binary image
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    thresh_kernel = max(int(round(char_length)), 1) if char_length else 21
    thresh_kernel = thresh_kernel + 1 if thresh_kernel % 2 == 0 else thresh_kernel

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_kernel, 5)
    binary_thresh = cv2.adaptiveThreshold(255 - blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_kernel, 5)

    # Mask on areas with dark background
    blur_size = min(255, max(int(2 * char_length) + 1 - int(2 * char_length) % 2, 1) if char_length else 11)
    blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    mask = cv2.inRange(blur, 0, 100)

    # Get contours of dark areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each dark area, use binary threshold instead of regular threshold
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        margin = int(char_length) if char_length else 21
        if min(w, h) > 2 * margin and w * h / np.prod(img.shape[:2]) < 0.9:
            thresh[y+margin:y+h-margin, x+margin:x+w-margin] = binary_thresh[y+margin:y+h-margin, x+margin:x+w-margin]

    return thresh


def dilate_dotted_lines(thresh: np.ndarray, char_length: float, contours: List[Cell]) -> np.ndarray:
    """
    Dilate specific rows/columns of the threshold image in order to detect dotted rows
    :param thresh: threshold image array
    :param char_length: average character length in image
    :param contours: list of image contours as cell objects
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

    # Filter clusters with contours
    filtered_rows_cl = list()
    for row_cl in white_rows_cl:
        # Compute percentage of white pixels in rows
        pct_w_pixels = np.mean(thresh[row_cl, :]) / 255
        # Compute percentage of rows covered by contours
        covered_contours = [cnt for cnt in contours if min(cnt.y2, max(row_cl)) - max(cnt.y1, min(row_cl)) > 0]
        pct_contours = sum(map(lambda cnt: cnt.width, covered_contours)) / thresh.shape[1]

        if 0.66 * pct_w_pixels >= pct_contours:
            filtered_rows_cl.append(row_cl)

    white_rows_final = [idx for cl in filtered_rows_cl for idx in cl]

    # Keep only dilated image on specific rows
    mask = np.ones(thresh.shape[0], dtype=bool)
    mask[white_rows_final] = False
    h_dilated[mask, :] = 0

    ### Vertical case
    # Create dilated image
    v_dilated = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(char_length), 1))))

    # Get columns with at least 2 times the average number of white pixels
    v_non_null = np.where(np.max(thresh, axis=0) > 0)[0]
    white_cols = np.where(np.mean(thresh[min(v_non_null):max(v_non_null), :], axis=0) > 4 * w_mean)[0].tolist()

    # Split into consecutive groups of columns and keep only small ones to avoid targeting text columns
    white_cols_cl = [list(map(itemgetter(1), g))
                     for k, g in groupby(enumerate(white_cols), lambda i_x: i_x[0] - i_x[1])]

    # Filter clusters with contours
    filtered_cols_cl = list()
    for col_cl in white_cols_cl:
        # Compute percentage of white pixels in columns
        pct_w_pixels = np.mean(thresh[:, col_cl]) / 255
        # Compute percentage of columns covered by contours
        covered_contours = [cnt for cnt in contours if min(cnt.x2, max(col_cl)) - max(cnt.x1, min(col_cl)) > 0]
        pct_contours = sum(map(lambda cnt: cnt.height, covered_contours)) / thresh.shape[0]

        if 0.66 * pct_w_pixels >= pct_contours:
            filtered_cols_cl.append(col_cl)

    white_cols_final = [idx for cl in filtered_cols_cl for idx in cl]

    # Keep only dilated image on specific columns
    mask = np.ones(thresh.shape[1], dtype=bool)
    mask[white_cols_final] = False
    v_dilated[:, mask] = 0

    # Update thresh
    new_thresh = np.maximum(thresh, h_dilated)
    new_thresh = np.maximum(new_thresh, v_dilated)

    return new_thresh


def overlapping_filter(lines: List[Line], max_gap: int = 5) -> List[Line]:
    """
    Process rows to merge close rows
    :param lines: rows
    :param max_gap: maximum gap used to merge rows
    :return: list of filtered rows
    """
    if len(lines) == 0:
        return []

    # Identify if rows are horizontal
    horizontal = np.average([l.horizontal for l in lines], weights=[l.length for l in lines]) > 0.5

    # If not horizontal, transpose all rows
    if not horizontal:
        lines = [line.transpose for line in lines]

    # Sort rows by secondary dimension
    lines = sorted(lines, key=lambda l: (l.y1, l.x1))

    # Create clusters of rows based on "similar" secondary dimension
    previous_sequence, current_sequence = iter(lines), iter(lines)
    line_clusters = [[next(current_sequence)]]
    for previous, line in zip(previous_sequence, current_sequence):
        # If the vertical difference between consecutive rows is too large, create a new cluster
        if line.y1 - previous.y1 > 2:
            # Large gap, we create a new empty sublist
            line_clusters.append([])

        # Append to last cluster
        line_clusters[-1].append(line)

    # Create final rows by "merging" rows within a cluster
    final_lines = list()
    for cluster in line_clusters:
        # Sort the cluster
        cluster = sorted(cluster, key=lambda l: min(l.x1, l.x2))

        # Loop over rows in the cluster to merge relevant rows together
        seq = iter(cluster)
        sub_clusters = [[next(seq)]]
        for line in seq:
            # If rows are vertically close, merge line with curr_line
            dim_2_sub_clust = max(map(lambda l: l.x2, sub_clusters[-1]))
            if line.x1 - dim_2_sub_clust <= max_gap:
                sub_clusters[-1].append(line)
            # If the difference in vertical coordinates is too large, create a new sub cluster
            else:
                sub_clusters.append([line])

        # Create rows from sub clusters
        for sub_cl in sub_clusters:
            y_value = int(round(np.average([l.y1 for l in sub_cl],
                                           weights=list(map(lambda l: l.length, sub_cl)))))
            thickness = min(max(1, max(map(lambda l: l.y2, sub_cl)) - min(map(lambda l: l.y1, sub_cl))), 5)
            line = Line(x1=min(map(lambda l: l.x1, sub_cl)),
                        x2=max(map(lambda l: l.x2, sub_cl)),
                        y1=int(y_value),
                        y2=int(y_value),
                        thickness=thickness)

            if line.length > 0:
                final_lines.append(line)

    # If not horizontal, transpose all rows
    if not horizontal:
        final_lines = [line.transpose for line in final_lines]

    return final_lines


def create_lines_from_intersection(line_dict: Dict) -> List[Line]:
    """
    Create list of lines from detected line and its intersecting elements
    :param line_dict: dictionary containing line and its intersecting elements
    :return: list of relevant line objects
    """
    # Get intersection segments
    inter_segs = [(inter_cnt.get('y1'), inter_cnt.get('y2')) if line_dict.get('vertical')
                  else (inter_cnt.get('x1'), inter_cnt.get('x2'))
                  for inter_cnt in line_dict.get('intersecting') or []
                  ]

    if len(inter_segs) == 0:
        # If no elements intersect the line, return it
        return [Line(x1=line_dict.get('x1_line'),
                     x2=line_dict.get('x2_line'),
                     y1=line_dict.get('y1_line'),
                     y2=line_dict.get('y2_line'),
                     thickness=line_dict.get('thickness'))
                ]
    
    # Vertical case
    if line_dict.get('vertical'):
        # Get x and y values of the line
        x, y_min, y_max = line_dict.get('x1_line'), line_dict.get('y1_line'), line_dict.get('y2_line')
        # Create y range of the line
        y_range = list(range(y_min, y_max + 1))

        # For each intersecting elements, remove common coordinates with the line
        for inter_seg in inter_segs:
            y_range = [y for y in y_range if not inter_seg[0] <= y <= inter_seg[1]]

        if y_range:
            # Create list of lists of consecutive y values from the range
            seq = iter(y_range)
            line_y_gps = [[next(seq)]]
            for y in seq:
                if y > line_y_gps[-1][-1] + 1:
                    line_y_gps.append([])
                line_y_gps[-1].append(y)

            return [Line(x1=x, x2=x, y1=min(y_gp), y2=max(y_gp), thickness=line_dict.get('thickness'))
                    for y_gp in line_y_gps]
        return []
    # Horizontal case
    else:
        # Get x and y values of the line
        y, x_min, x_max = line_dict.get('y1_line'), line_dict.get('x1_line'), line_dict.get('x2_line')
        # Create x range of the line
        x_range = list(range(x_min, x_max + 1))

        # For each intersecting elements, remove common coordinates with the line
        for inter_seg in inter_segs:
            x_range = [x for x in x_range if not inter_seg[0] <= x <= inter_seg[1]]

        if x_range:
            # Create list of lists of consecutive x values from the range
            seq = iter(x_range)
            line_x_gps = [[next(seq)]]
            for x in seq:
                if x > line_x_gps[-1][-1] + 1:
                    line_x_gps.append([])
                line_x_gps[-1].append(x)

            return [Line(y1=y, y2=y, x1=min(x_gp), x2=max(x_gp), thickness=line_dict.get('thickness'))
                    for x_gp in line_x_gps]
        return []


def remove_word_lines(lines: List[Line], contours: List[Cell]) -> List[Line]:
    """
    Remove rows that corresponds to contours in image
    :param lines: list of rows
    :param contours: list of image contours as cell objects
    :return: list of rows not intersecting with words
    """
    # If there are no rows or no contours, do nothing
    if len(lines) == 0 or len(contours) == 0:
        return lines

    # Get contours dataframe
    df_cnts = pl.LazyFrame(data=[{"x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2} for c in contours])

    # Create dataframe containing rows
    df_lines = (pl.LazyFrame(data=[{**line.dict, **{"id_line": idx}} for idx, line in enumerate(lines)])
                .with_columns([pl.max_horizontal([pl.col('width'), pl.col('height')]).alias('length'),
                               (pl.col('x1') == pl.col('x2')).alias('vertical')]
                              )
                .rename({"x1": "x1_line", "x2": "x2_line", "y1": "y1_line", "y2": "y2_line"})
                )

    # Merge both dataframes
    df_words_lines = df_cnts.join(df_lines, how='cross')

    # Compute intersection between contours bbox and rows
    # - vertical case
    vert_int = (
            (((pl.col('x1') + pl.col('x2')) / 2 - pl.col('x1_line')).abs() / (pl.col('x2') - pl.col('x1')) < 0.5)
            & ((pl.min_horizontal(['y2', 'y2_line']) - pl.max_horizontal(['y1', 'y1_line'])) > 0)
    )
    # - horizontal case
    hor_int = (
            (((pl.col('y1') + pl.col('y2')) / 2 - pl.col('y1_line')).abs() / (pl.col('y2') - pl.col('y1')) <= 0.4)
            & ((pl.min_horizontal(['x2', 'x2_line']) - pl.max_horizontal(['x1', 'x1_line'])) > 0)
    )
    
    df_words_lines = df_words_lines.with_columns(
        ((pl.col('vertical') & vert_int) | ((~pl.col('vertical')) & hor_int)).alias('intersection')
        )
    
    # Get lines together with elements that intersect the line
    line_elements = (df_words_lines.filter(pl.col('intersection'))
                     .group_by(["id_line", "x1_line", "y1_line", "x2_line", "y2_line", "vertical", "thickness"])
                     .agg(pl.struct("x1", "y1", "x2", "y2").alias('intersecting'))
                     .unique(subset=["id_line"])
                     .collect()
                     .to_dicts()
                     )

    # Create lines from line elements
    modified_lines = {el.get('id_line') for el in line_elements}
    kept_lines = [line for id_line, line in enumerate(lines) if id_line not in modified_lines]
    reprocessed_lines = [line for line_dict in line_elements
                         for line in create_lines_from_intersection(line_dict=line_dict)]

    return kept_lines + reprocessed_lines


def detect_lines(thresh: np.ndarray, contours: Optional[List[Cell]], char_length: Optional[float], rho: float = 1,
                 theta: float = np.pi / 180, threshold: int = 50, minLinLength: int = 290, maxLineGap: int = 6,
                 kernel_size: int = 20) -> (List[Line], List[Line]):
    """
    Detect horizontal and vertical rows on image
    :param thresh: thresholded image array
    :param contours: list of image contours as cell objects
    :param char_length: average character length
    :param rho: rho parameter for Hough line transform
    :param theta: theta parameter for Hough line transform
    :param threshold: threshold parameter for Hough line transform
    :param minLinLength: minLinLength parameter for Hough line transform
    :param maxLineGap: maxLineGap parameter for Hough line transform
    :param kernel_size: kernel size to filter on horizontal / vertical rows
    :return: horizontal and vertical rows
    """
    if char_length is not None:
        # Process threshold image in order to detect dotted rows
        thresh = dilate_dotted_lines(thresh=thresh, char_length=char_length, contours=contours)

    # Identify both vertical and horizontal rows
    for kernel_tup, gap in [((kernel_size, 1), 2 * maxLineGap), ((1, kernel_size), maxLineGap)]:
        # Apply masking on image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_tup)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Compute Hough rows on image and get rows
        hough_lines = cv2.HoughLinesP(mask, rho, theta, threshold, None, minLinLength, maxLineGap)

        # Handle case with no rows
        if hough_lines is None:
            yield []
            continue

        lines = [Line(*line[0].tolist()).reprocess() for line in hough_lines]

        # Remove rows that are not horizontal or vertical
        lines = [line for line in lines if line.horizontal or line.vertical]

        # Merge rows
        merged_lines = overlapping_filter(lines=lines, max_gap=gap)

        # If possible, remove rows that correspond to words
        if contours is not None:
            merged_lines = remove_word_lines(lines=merged_lines, contours=contours)
            merged_lines = [l for l in merged_lines if max(l.length, l.width) >= minLinLength]

        yield merged_lines
