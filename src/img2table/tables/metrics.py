import cv2
import numpy as np

from img2table.tables._metrics import (  # ty:ignore[unresolved-import]
    compute_contours,
    create_character_thresh,
    filter_cc,
    get_row_separations,
    identify_obstacles,
    remove_dots,
    remove_dotted_lines,
)
from img2table.tables.types import Cell


def compute_char_length(
    thresh: np.ndarray,
) -> tuple[float | None, np.ndarray | None]:
    """
    Compute average character length based on connected components' analysis
    :param thresh: threshold image array
    :return: tuple with average character length, thresholded image of characters and array of image characters
    """
    # Connected components
    _, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
        image=thresh, connectivity=8, ltype=cv2.CV_32S
    )

    # Remove dots
    stats = remove_dots(cc_labels=cc_labels, stats=stats)

    # Remove connected components with less than 10 pixels
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 5
    stats = stats[mask_pixels]

    if len(stats) == 0:
        return None, None

    # Remove dotted lines
    complete_stats = np.c_[
        stats, (2 * stats[:, 0] + stats[:, 2]) / 2, (2 * stats[:, 1] + stats[:, 3]) / 2
    ]
    stats = remove_dotted_lines(complete_stats=complete_stats)

    if len(stats) == 0:
        return None, None

    # Filter relevant connected components
    relevant_stats, discarded_stats = filter_cc(stats=stats)

    if len(relevant_stats) > 0:
        # Compute average character length
        argmax_char_length = float(np.argmax(np.bincount(relevant_stats[:, cv2.CC_STAT_WIDTH])))
        mean_char_length = np.mean(relevant_stats[:, cv2.CC_STAT_WIDTH])
        char_length = (
            mean_char_length if 1.5 * argmax_char_length <= mean_char_length else argmax_char_length
        )

        # Create thresholded image with characters
        _, chars_array = create_character_thresh(
            thresh=thresh,
            stats=relevant_stats,
            discarded_stats=discarded_stats,
            char_length=char_length,
        )

        return char_length, chars_array
    return None, None

def compute_median_line_sep(
    chars_array: np.ndarray, char_length: float, height: int, width: int
) -> tuple[float | None, list[Cell] | None]:
    """
    Compute median separation between rows
    :param chars_array: array of characters
    :param char_length: average character length
    :param height: image height
    :param width: image width
    :return: median separation between rows
    """
    # Identify obstacles
    vertical_obstacles = identify_obstacles(
        stats=chars_array,
        min_width=char_length / 3,
        min_height=height / 5,
        height=height,
        width=width,
    )
    horizontal_obstacles = identify_obstacles(
        stats=chars_array,
        min_width=width / 5,
        min_height=char_length / 4,
        height=height,
        width=width,
    )
    obstacles = np.maximum(vertical_obstacles, horizontal_obstacles)

    # Identify merged contours
    stats_contours = compute_contours(
        chars_array=chars_array, obstacles=obstacles, char_length=char_length
    )

    # Compute median line sep
    row_separations = get_row_separations(stats=stats_contours, char_length=char_length)

    if row_separations.size > 0:
        # Bin separations and compute most common separation
        separations = 2 * np.floor(np.asarray(row_separations, dtype=np.float64) / 2) + 1
        unique_separations, counts = np.unique(separations, return_counts=True)
        idx = np.lexsort((unique_separations, -counts))
        median_line_sep = float(unique_separations[idx][0])
    else:
        median_line_sep = None

    # Get contours cells
    contours_cells = [Cell(x1=x, y1=y, x2=x + w, y2=y + h) for x, y, w, h in stats_contours]

    return median_line_sep, contours_cells


def compute_img_metrics(
    thresh: np.ndarray,
) -> tuple[float | None, float | None, list[Cell] | None]:
    """
    Compute metrics from image
    :param thresh: threshold image array
    :return: average character length, median line separation and image contours
    """
    # Compute average character length based on connected components analysis
    char_length, chars_array = compute_char_length(thresh=thresh)

    if char_length is None or chars_array is None:
        return None, None, None

    # Compute median separation between rows
    median_line_sep, contours = compute_median_line_sep(
        chars_array=chars_array,
        char_length=char_length,
        height=thresh.shape[0],
        width=thresh.shape[1],
    )

    return char_length, median_line_sep, contours
