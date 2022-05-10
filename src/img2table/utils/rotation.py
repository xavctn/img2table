# coding: utf-8
import math
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytesseract
from cv2 import cv2

from img2table.objects.tables import Line


def rotate(image: np.ndarray, angle: float, background: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Rotate image
    :param image: image
    :param angle: angle
    :param background: background color for rotated image
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


def img_to_horizontal_lines(image: np.ndarray) -> np.ndarray:
    """
    Detect if the image is tilted by 90° and correct it if needed
    :param image: image array
    :return: image with horizontal text (can be flipped upside down)
    """
    img = image.copy()
    # Image to gray and canny
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(img, 50, 200, None, 3)

    # Compute Hough lines on image
    lines = cv2.HoughLinesP(dst, 0.5, np.pi / 180, 10, None, 20, 10)
    lines = [Line(line=line[0]) for line in lines]

    # If no lines found, return image
    if len(lines) == 0:
        return image

    # Create clusters of lines
    df_lines = pd.DataFrame([{"angle": line.angle, "length": line.length, "angle_length": line.angle * line.length}
                             for line in lines])
    df_lines = df_lines.sort_values(by=['angle'])
    df_lines["cluster"] = (df_lines["angle"] - df_lines["angle"].shift() >= 0.5).astype(int).cumsum()

    # Get angle of the most represented cluster
    df_clusters_angle = (df_lines.groupby('cluster')
                         .agg(length=('length', np.sum),
                              angle_length=('angle_length', np.sum))
                         )
    df_clusters_angle["angle"] = df_clusters_angle['angle_length'] / df_clusters_angle['length']

    angle = df_clusters_angle.sort_values('length', ascending=False).iloc[0, 2]

    # Rotate image according to the angle
    if angle is not None:
        if abs(angle) > 1:
            return rotate(image=image, angle=angle)

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
        img_rotated = rotate(horizontal_img, angle=angle)
    else:
        img_rotated = horizontal_img

    return img_rotated
