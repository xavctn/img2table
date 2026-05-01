"""
ARLSA pipeline based on:
Nikolaou et al., "A segmentation framework for historical machine-printed documents",
Image and Vision Computing 28 (2010) 590-604.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from img2table.tables.borderless.layout._text_lines import (  # ty:ignore[unresolved-import]
    _remove_punctuation_marks,
    adaptive_rlsa,
    detect_obstacles,
    remove_noise,
)
from img2table.tables.common import find_components
from img2table.tables.types import Cell

if TYPE_CHECKING:
    from img2table.tables.types import Line, Table


def _mask_detected_lines(
    thresh: np.ndarray, lines: list[Line] | None, char_length: float | None
) -> np.ndarray:
    """
    Remove lines from the thresholded image.
    :param thresh: threshold image array
    :param lines: list of image rows
    :param char_length: average character length
    :return: thresholded image with lines masked out
    """
    if lines is None or char_length is None:
        return thresh

    for line in lines:
        if line.horizontal and line.length >= 3 * char_length:
            cv2.rectangle(
                thresh,
                (line.x1, line.y1 - line.thickness // 2 - 1),
                (line.x2, line.y2 + line.thickness // 2 + 1),
                0,
                -1,
            )
        elif line.vertical and line.length >= 2 * char_length:
            cv2.rectangle(
                thresh,
                (line.x1 - line.thickness // 2 - 1, line.y1),
                (line.x2 + line.thickness // 2 + 1, line.y2),
                0,
                -1,
            )

    return thresh


def _apply_arlsa(img: np.ndarray, a: float, obstacle_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Apply ARLSA to a binary image and return a binary image.
    :param img: input binary image
    :param a: connected components' distance ratio
    :param obstacle_mask: obstacle mask array
    :return: RLSA resulting image
    """
    if obstacle_mask is None:
        obstacle_mask = np.zeros(shape=img.shape, dtype=np.uint8)

    _, cc, cc_stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)
    if len(cc_stats) <= 1:
        return img.copy()

    rlsa = adaptive_rlsa(
        cc=cc,
        cc_stats=cc_stats,
        obstacle_mask=obstacle_mask,
        a=float(a),
        th=3.5,
        c=0.4,
    )
    return (rlsa > 0).astype(np.uint8) * 255


def remove_punctuation_marks(thresh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove punctuation marks from a binary image using ARLSA
    :param thresh: input threshold image
    :return: input image with punctuation removee
    """
    # Apply RLSA
    rlsa = _apply_arlsa(img=thresh.copy(), a=1.5)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(rlsa, connectivity=8, ltype=cv2.CV_32S)

    # Create cleaned and punctuation images
    return _remove_punctuation_marks(thresh=thresh, labels=labels, stats=stats)


def identify_text_mask(
    thresh: np.ndarray,
    lines: list[Line],
    char_length: float,
    existing_tables: list[Table] | None = None,
) -> np.ndarray:
    """
    Identify text mask of the input image
    :param thresh: threshold image array
    :param lines: list of image rows
    :param char_length: average character length
    :param existing_tables: list of detected bordered tables
    :return: thresholded image corresponding to the text mask
    """
    # Remove lines from thresholded image
    thresh = _mask_detected_lines(thresh=thresh, lines=lines, char_length=char_length)

    # Compute connected components
    _, cc, cc_stats, _ = cv2.connectedComponentsWithStats(
        image=thresh, connectivity=8, ltype=cv2.CV_32S
    )
    if len(cc_stats) <= 1:
        return thresh

    average_height = float(np.mean(cc_stats[1:, cv2.CC_STAT_HEIGHT]))  # ty: ignore[no-matching-overload]
    median_width = float(np.median(cc_stats[1:, cv2.CC_STAT_WIDTH]))  # ty: ignore[no-matching-overload]
    cc_denoised = remove_noise(
        cc=cc.copy(), cc_stats=cc_stats, average_height=average_height, median_width=median_width
    )
    denoised = (cc_denoised > 0).astype(np.uint8) * 255

    # Remove punctuation
    cleaned, punctuation = remove_punctuation_marks(thresh=denoised)

    # Stage 3: obstacle detection on ARLSA(a=1.5) result
    obstacle_input = _apply_arlsa(cleaned, a=0.75)
    obstacles = detect_obstacles(img=obstacle_input, min_width=char_length)

    # Identify text lines and put back punctuation in final mask
    text_lines = _apply_arlsa(img=cleaned, a=4.0, obstacle_mask=obstacles)
    final_mask = np.maximum(text_lines, punctuation)

    # Remove elements from existing table positions
    for tb in existing_tables or []:
        final_mask[tb.y1 : tb.y2, tb.x1 : tb.x2] = 0

    return final_mask


def regroup_contours(cnts: list[Cell], char_length: float) -> list[Cell]:
    """
    Merge close contours together
    :param cnts: list of contours
    :param char_length: average character length
    :return: list of grouped contours
    """
    cnts = sorted(cnts, key=lambda cnt: (cnt.y1, cnt.x1))

    # Identify matching contours
    edges = []
    for i, cnt1 in enumerate(cnts):
        for j in range(i, len(cnts)):
            cnt2 = cnts[j]

            if cnt2.y1 >= cnt1.y2:
                break

            # Compute y overlap and check correspondence
            y_overlap = min(cnt1.y2, cnt2.y2) - max(cnt1.y1, cnt2.y1)
            if y_overlap < 0.25 * min(cnt1.height, cnt2.height):
                continue

            # Compute x overlap / distance and check correspondence
            x_overlap = min(cnt1.x2, cnt2.x2) - max(cnt1.x1, cnt2.x1)
            if x_overlap < -char_length:
                continue

            edges.append({i, j})

    # Identify groups
    contour_groups = find_components(edges=edges)

    return [
        Cell(
            x1=min(cnts[idx].x1 for idx in gp),
            y1=min(cnts[idx].y1 for idx in gp),
            x2=max(cnts[idx].x2 for idx in gp),
            y2=max(cnts[idx].y2 for idx in gp),
        )
        for gp in contour_groups
    ]


def identify_image_contours(
    thresh: np.ndarray,
    lines: list[Line],
    char_length: float,
    existing_tables: list[Table] | None = None,
) -> list[Cell]:
    """
    Identify contours of the input image
    :param thresh: threshold image array
    :param lines: list of image rows
    :param char_length: average character length
    :param existing_tables: list of detected bordered tables
    :return: list of contours identified as cells
    """
    # Get text mask
    text_mask = identify_text_mask(
        thresh=thresh, lines=lines, char_length=char_length, existing_tables=existing_tables
    )

    # Find contours, highlight text areas, and extract ROIs
    cnts, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Regroup raw contours first so nearby dots can merge into their text contour.
    contours = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        contours.append(Cell(x1=x, y1=y, x2=x + w, y2=y + h))

    grouped_contours = regroup_contours(cnts=contours, char_length=char_length)

    return [
        cnt
        for cnt in grouped_contours
        if min(cnt.height, cnt.width) >= 0.5 * char_length
        and max(cnt.height, cnt.width) >= char_length
    ]
