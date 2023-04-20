# coding: utf-8
from typing import Tuple, List

import cv2
import numpy as np
import polars as pl

dixon_q_test_confidence_dict = {
    0.9: {3: 0.941, 4: 0.765, 5: 0.642, 6: 0.56, 7: 0.507, 8: 0.468, 9: 0.437, 10: 0.412},
    0.95: {3: 0.970, 4: 0.829, 5: 0.71, 6: 0.625, 7: 0.568, 8: 0.526, 9: 0.493, 10: 0.466},
    0.99: {3: 0.994, 4: 0.926, 5: 0.821, 6: 0.74, 7: 0.68, 8: 0.634, 9: 0.598, 10: 0.568}
}


def get_connected_components(img: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Identify connected components in image
    :param img: image array
    :return: list of connected components centroids and thresholded image
    """
    # Thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components
    _, _, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # Remove connected components with less than 15 pixels
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 15
    stats = stats[mask_pixels]

    # Compute median width and height
    median_width = np.median(stats[:, cv2.CC_STAT_WIDTH])
    median_height = np.median(stats[:, cv2.CC_STAT_HEIGHT])

    # Compute bbox area bounds
    upper_bound = 4 * median_width * median_height
    lower_bound = 0.25 * median_width * median_height

    # Filter connected components according to their area
    mask_lower_area = lower_bound < stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_area = upper_bound > stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_area = mask_lower_area & mask_upper_area

    # Filter components based on aspect ratio
    mask_lower_ar = 0.5 < stats[:, cv2.CC_STAT_WIDTH] / stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_ar = 2 > stats[:, cv2.CC_STAT_WIDTH] / stats[:, cv2.CC_STAT_HEIGHT]
    mask_ar = mask_lower_ar & mask_upper_ar

    # Create general mask
    mask = mask_area & mask_ar

    # Get centroids from mask
    stats = stats[mask]
    centroids_x = stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] / 2
    centroids_y = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] / 2
    filtered_centroids = np.column_stack([centroids_x, centroids_y])

    return filtered_centroids, median_height, thresh


def get_relevant_angles(centroids: np.ndarray, ref_height: float, n_max: int = 5) -> List[float]:
    """
    Identify relevant angles from connected components centroids
    :param centroids: array of connected components centroids
    :param ref_height: reference height
    :param n_max: maximum number of returned angles
    :return: list of angle values
    """
    # Create dataframe with centroids
    df_centroids = pl.LazyFrame(data=centroids, schema=['x1', 'y1'])

    # Cross join and keep only relevant pairs
    df_cross = (df_centroids.join(df_centroids, how='cross')
                .filter(pl.col('x1') != pl.col('x1_right'))
                .filter((pl.col('y1') - pl.col('y1_right')).abs() <= 10 * ref_height)
                )

    # Compute slopes and angles
    df_angles = (df_cross.with_columns(((pl.col('y1') - pl.col('y1_right')) / (pl.col('x1') - pl.col('x1_right'))
                                        ).alias('slope'))
                 .with_columns((pl.col('slope').arctan() * 180 / np.pi).alias('angle'))
                 .with_columns(pl.when(pl.col('angle').abs() <= 45)
                               .then(pl.col('angle'))
                               .otherwise(pl.min(pl.col('angle') + 90, 90 - pl.col('angle')) * -pl.col('angle').sign())
                               .alias('angle')
                               )
                 )

    # Get n most represented angles
    most_likely_angles = (df_angles.groupby('angle')
                          .count()
                          .sort(by=['count', pl.col('angle').abs()], descending=[True, False])
                          .limit(n_max)
                          .collect(streaming=True)
                          .to_dicts()
                          )

    return sorted(list(set([angle.get('angle') for angle in most_likely_angles
                            if angle.get('count') >= 0.25 * max([a.get('count') for a in most_likely_angles])])))


def angle_dixon_q_test(angles: List[float], confidence: float = 0.9) -> float:
    """
    Compute best angle according to Dixon Q test
    :param angles: list of possible angles
    :param confidence: confidence level for outliers (0.9, 0.95, 0.99)
    :return: estimated angle
    """
    # Get dict of Q crit corresponding to confidence level
    dict_q_crit = dixon_q_test_confidence_dict.get(confidence)

    while len(angles) >= 3:
        # Compute range
        rng = angles[-1] - angles[0]

        # Get outlier and compute diff with closest angle
        diffs = [abs(nexxt - prev) for prev, nexxt in zip(angles, angles[1:])]
        idx_outlier = 0 if np.argmax(diffs) == 0 else len(angles) - 1
        gap = np.max(diffs)

        # Compute Qexp and compare to Qcrit
        q_exp = gap / rng

        if q_exp > dict_q_crit.get(len(angles)):
            angles.pop(idx_outlier)
        else:
            break

    return np.mean(angles)


def rotate_img(img: np.ndarray, angle: float) -> np.array:
    """
    Rotate image by angle
    :param img: image array
    :param angle: rotation angle
    :return: rotated image
    """
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def evaluate_angle(img: np.ndarray, angle: float) -> int:
    """
    Evaluate relevance of angle for image rotation
    :param img: image array
    :param angle: angle
    :return: metric for angle quality
    """
    # Rotate image
    rotated_img = rotate_img(img=img, angle=angle)
    # Apply horizontal projection
    proj = np.sum(rotated_img, 1)
    # Count number of empty rows
    return np.sum((proj[1:] - proj[:-1]) ** 2)


def estimate_skew(angles: List[float], thresh: np.ndarray) -> float:
    """
    Estimate skew from angles
    :param angles: list of angles
    :param thresh: thresholded image
    :return: best angle
    """
    # If there is only one angle, return it
    if len(angles) == 1:
        return angles.pop()

    if angles[-1] - angles[0] <= 0.015:
        # Get angle by applying Dixon Q test
        best_angle = angle_dixon_q_test(angles=angles)
    else:
        # Evaluate angles by rotation
        best_angle = None
        best_evaluation = 0
        for angle in sorted(angles, key=lambda a: abs(a)):
            # Get angle evaluation
            angle_evaluation = evaluate_angle(img=thresh, angle=angle)

            if angle_evaluation > best_evaluation:
                best_angle = angle
                best_evaluation = angle_evaluation

    return best_angle or 0


def rotate_img_with_border(img: np.ndarray, angle: float,
                           background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Rotate an image of the defined angle and add background on border
    :param img: image array
    :param angle: rotation angle
    :param background_color: background color for borders after rotation
    :return: rotated image array
    """
    # Compute image center
    height, width = img.shape
    image_center = (width // 2, height // 2)

    # Compute rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # Get rotated image dimension
    bound_w = int(height * abs(rotation_mat[0, 1]) + width * abs(rotation_mat[0, 0]))
    bound_h = int(height * abs(rotation_mat[0, 0]) + width * abs(rotation_mat[0, 1]))

    # Update rotation matrix
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # Create rotated image with white background
    rotated_img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=background_color)
    return rotated_img


def fix_rotation_image(img: np.ndarray) -> np.ndarray:
    """
    Fix rotation of input image (based on https://www.mdpi.com/2079-9292/9/1/55) by at most 45 degrees
    :param img: image array
    :return: rotated image array
    """
    # Get connected components of the images
    cc_centroids, ref_height, thresh = get_connected_components(img=img)

    # Check number of centroids
    if len(cc_centroids) < 2:
        return img

    # Compute most likely angles from connected components
    angles = get_relevant_angles(centroids=cc_centroids, ref_height=ref_height)
    # Estimate skew
    skew_angle = estimate_skew(angles=angles, thresh=thresh)

    if skew_angle != 0:
        # Rotate image with borders
        return rotate_img_with_border(img=img, angle=skew_angle)

    return img
