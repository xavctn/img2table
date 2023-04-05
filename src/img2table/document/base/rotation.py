# coding: utf-8
import random
from collections import Counter
from typing import Tuple, List

import cv2
import numpy as np


def get_connected_components(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    return filtered_centroids, thresh


def get_relevant_slopes(centroids: np.ndarray, n_max: int = 10) -> List[float]:
    """
    Identify relevant slopes from connected components centroids
    :param centroids: array of connected components centroids
    :param n_max: maximum number of returned slopes
    :return: list of slopes values
    """
    nb_iter = 0
    slopes = list()
    while nb_iter < 30 * len(centroids):
        nb_iter += 1
        idx1, idx2 = random.sample(range(len(centroids)), 2)

        # Compute slope
        slope = (centroids[idx1, 1] - centroids[idx2, 1]) / (centroids[idx1, 0] - centroids[idx2, 0])
        slopes.append(slope)

    # Get n_max most frequent slopes
    most_frequent_slopes = [slope[0] for slope in Counter(slopes).most_common()[:n_max]]

    return most_frequent_slopes


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
    return np.sum(proj == 0)


def estimate_skew(slopes: List[float], thresh: np.ndarray) -> float:
    """
    Estimate skew from slopes
    :param slopes: list of slopes
    :param thresh: thresholded image
    :return: best angle
    """
    # Get angles from slopes
    angles = [np.arctan(slope) * 180 / np.pi for slope in slopes]

    angles = [angle if abs(angle) <= 45 else min(angle + 90, 90 - angle) * -np.sign(angle)
              for angle in angles]

    # Evaluate angles
    best_angle = None
    best_evaluation = 0
    for angle in angles:
        # Get angle evaluation
        angle_evaluation = evaluate_angle(img=thresh, angle=angle)

        if angle_evaluation > best_evaluation:
            best_angle = angle if abs(angle) < 45 else 90 - abs(angle)
            best_evaluation = angle_evaluation

    return best_angle or 0


def rotate_img_with_border(img: np.ndarray, angle: float, background_color: Tuple[int] = (255, 255, 255)) -> np.ndarray:
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
    cc_centroids, thresh = get_connected_components(img=img)

    # Compute most likely slopes from connected components
    slopes = get_relevant_slopes(centroids=cc_centroids)

    # Estimate skew
    skew_angle = estimate_skew(slopes=slopes, thresh=thresh)

    if skew_angle != 0:
        # Rotate image with borders
        return rotate_img_with_border(img=img, angle=skew_angle)

    return img
