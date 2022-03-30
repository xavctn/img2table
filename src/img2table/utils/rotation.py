# coding: utf-8
import math
import os
import re
import shutil
import tempfile
from typing import List, Iterable

import numpy as np
import pytesseract
from cv2 import cv2

from img2table.objects.tables import Line


def rotate(image: np.ndarray, angle: float, background) -> np.ndarray:
    """
    Rotate image
    :param image: image
    :param angle: angle
    :param background:
    :return: rotated image
    """
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def line_cluster(lines: List[Line], angle_delta: float = 1.0) -> Iterable[List[Line]]:
    """
    Create iterable of line clusters based on line angle
    :param lines:
    :param angle_delta:
    :return:
    """
    # Order lines by angle
    lines = sorted(lines, key=lambda line: line.angle)

    # Create clusters of line angles
    group = list()
    for idx, line in enumerate(lines):
        if idx == 0:
            group = [line]
        elif line.angle - group[-1].angle <= angle_delta:
            group.append(line)
        else:
            yield group
            group = [line]

    if group:
        yield group


def img_to_horizontal_lines(image: np.ndarray) -> np.ndarray:
    """
    Detect if the image is tilted by 90° and correct it if needed
    :param image: image array
    :return: image with horizontal text (can be flipped upside down)
    """
    img = image.copy()
    # Image to gray and canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(gray, 50, 200, None, 3)

    # Compute Hough lines on image
    lines = cv2.HoughLinesP(dst, 0.5, np.pi / 180, 5, None, 20, 10)
    lines = [Line(line=line[0]) for line in lines]

    # Get image angle
    max_length = 0
    angle = None
    for cluster in line_cluster(lines):
        group_length = sum([line.length for line in cluster])
        if group_length > max_length:
            max_length = group_length
            angle = sum([line.angle * line.length for line in cluster]) / group_length

    # Rotate image according to the angle
    if angle is not None:
        if abs(angle) > 1:
            return rotate(image, angle, (255, 255, 255))

    return image


def get_orientation_image(img: np.ndarray) -> int:
    """
    Detect orientation angle of an image
    :param img: image
    :return: orientation angle of the image
    """
    # Create temporary path and write image
    dirpath = tempfile.mkdtemp()
    tmpfp = os.path.join(dirpath, "img.png")
    cv2.imwrite(tmpfp, img)

    # Get orientation
    osd = pytesseract.image_to_osd(tmpfp)

    # Delete temporary dir
    shutil.rmtree(dirpath)

    # Retrieve angle for rotation
    angle = 360 - int(re.search(r'(?<=Rotate: )\d+', osd).group(0))

    return angle


def rotate_img(img: np.ndarray) -> np.ndarray:
    """
    Correction on image rotation
    :param img: image array
    :return: image array with correct orientation
    """
    # Rotate image in order to get horizontal text lines
    horizontal_img = img_to_horizontal_lines(image=img)

    # Detect text orientation angle
    angle = get_orientation_image(horizontal_img)

    # Rotate image if needed
    if angle % 360 == 180:
        img_rotated = rotate(horizontal_img, angle=angle, background=(0, 0, 0))
    else:
        img_rotated = horizontal_img

    return img_rotated
