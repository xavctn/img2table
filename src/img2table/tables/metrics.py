import cv2
import numpy as np

from img2table.tables._metrics import (  # ty:ignore[unresolved-import]
    create_character_thresh,
    filter_cc,
    get_row_separations,
    recompute_contours,
    remove_dots,
    remove_dotted_lines,
)
from img2table.tables.types import Cell


def compute_char_length(
    thresh: np.ndarray,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
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
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 10
    stats = stats[mask_pixels]

    if len(stats) == 0:
        return None, None, None

    # Remove dotted lines
    complete_stats = np.c_[
        stats, (2 * stats[:, 0] + stats[:, 2]) / 2, (2 * stats[:, 1] + stats[:, 3]) / 2
    ]
    stats = remove_dotted_lines(complete_stats=complete_stats)

    if len(stats) == 0:
        return None, None, None

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
        characters_thresh, chars_array = create_character_thresh(
            thresh=thresh,
            stats=relevant_stats,
            discarded_stats=discarded_stats,
            char_length=char_length,
        )

        return char_length, characters_thresh, chars_array
    return None, None, None


def compute_median_line_sep(
    thresh_chars: np.ndarray, chars_array: np.ndarray, char_length: float
) -> tuple[float | None, list[Cell] | None]:
    """
    Compute median separation between rows
    :param thresh_chars: thresholded image of characters
    :param char_length: average character length
    :return: median separation between rows
    """
    # Identify characters that belong to the same word and create merged contours, by closing image and retrieving
    # connected components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(char_length // 2 + 1), 1))
    thresh_chars = cv2.morphologyEx(thresh_chars, cv2.MORPH_CLOSE, kernel)

    _, _, stats, _ = cv2.connectedComponentsWithStats(
        image=thresh_chars, connectivity=8, ltype=cv2.CV_32S
    )

    # Recompute contours
    stats_contours = recompute_contours(stats=stats, chars_array=chars_array)

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
    char_length, thresh_chars, chars_array = compute_char_length(thresh=thresh)

    if char_length is None or thresh_chars is None or chars_array is None:
        return None, None, None

    # Compute median separation between rows
    median_line_sep, contours = compute_median_line_sep(
        thresh_chars=thresh_chars, chars_array=chars_array, char_length=char_length
    )

    return char_length, median_line_sep, contours
