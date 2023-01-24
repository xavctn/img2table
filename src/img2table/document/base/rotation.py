# coding: utf-8
import math
from typing import Tuple

import cv2
import numpy as np


def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image of the defined angle
    :param img: image array
    :param angle: rotation angle
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
                                 borderValue=(255, 255, 255))
    return rotated_img


def straightened_img(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Straighten an image if it is skewed. Resulting image might be upside down.
    :param img: image array
    :return: tuple with straightened image array and angle value
    """
    # Get image edges
    edges = cv2.Canny(img, 100, 100, apertureSize=3)

    # Compute probabilistic lines with Hough Transform
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=math.pi / 180.0,
                            threshold=50,
                            minLineLength=100,
                            maxLineGap=20)

    # Get the median angle of detected lines
    median_angle = float(np.median([math.degrees(math.atan2(y2 - y1, x2 - x1)) for [[x1, y1, x2, y2]] in lines]))

    if median_angle % 180 != 0:
        # Rotate the image to straighten it
        straight_img = rotate_img(img=img, angle=median_angle)
        return straight_img, median_angle

    return img, 0.0


def upside_down(img: np.ndarray) -> bool:
    """
    Identify is an image is upside down
    :param img: image array
    :return: boolean indicating if the image is upside down
    """
    height, width = img.shape

    # For both the top and the bottom part of the image, compute the area before first black pixel
    top_area = 0
    bottom_area = 0

    for id_col in range(width):
        # Identify black pixels in the column
        black_pixels = np.where(img[:, id_col] == 0)[0]

        if black_pixels.size > 0:
            # Add column area to the totals
            top_area += np.amin(black_pixels)
            bottom_area += height - np.amax(black_pixels) - 1

    # Assumption is that the top part should contain more text than the bottom part --> area before first black pixels
    # should be lower in the top part than the one in the bottom part.
    return top_area > bottom_area


def fix_rotation_image(img: np.ndarray) -> np.ndarray:
    """
    Fix rotation of input image
    :param img: image array
    :return: rotated image array
    """
    # Apply filter to remove noise and thresholding
    denoised = cv2.medianBlur(img.copy(), 3)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Straighten image
    straight_thresh, angle = straightened_img(img=thresh)

    # Identify if the straightened image is upside down
    is_inverted = upside_down(img=straight_thresh)

    # Compute final rotation angle to apply
    rotation_angle = angle + 180 * int(is_inverted)

    if rotation_angle % 360 > 0:
        return rotate_img(img=img, angle=rotation_angle)

    return img
