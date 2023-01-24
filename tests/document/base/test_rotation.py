# coding: utf-8
import cv2
import numpy as np
from sewar import ssim

from img2table.document.base.rotation import straightened_img, rotate_img, upside_down, fix_rotation_image


def test_straightened_img():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    for angle in range(-180, 180, 1):
        # Create test image by rotating it
        test_img = rotate_img(img=img.copy(), angle=angle)
        _, rotation_angle = straightened_img(img=test_img)

        # Compute angle error
        error = min(abs(rotation_angle + angle),
                    abs(rotation_angle + angle + 180),
                    abs(rotation_angle + angle - 180))
        # Check if the error is less than half a degree
        assert error < 0.5


def test_upside_down():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    assert not upside_down(img=img)
    assert upside_down(img=rotate_img(img, 180))


def test_fix_rotation_image():
    def crop_to_orig_img(img, orig_img):
        # Get original dimensions
        orig_height, orig_width = orig_img.shape

        # Get center of img
        center = (img.shape[0] // 2, img.shape[1] // 2)
        # Crop img around centre
        cropped = img[center[0] - orig_height // 2: center[0] + orig_height // 2 + 1,
                      center[1] - orig_width // 2: center[1] + orig_width // 2 + 1]

        return cropped

    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    similarities = list()
    for angle in range(-180, 180, 3):
        # Create test image by rotating it
        test_img = rotate_img(img=img.copy(), angle=angle)
        result = crop_to_orig_img(img=fix_rotation_image(img=test_img),
                                  orig_img=img)

        # Compute similarity between original image and result
        similarities.append(ssim(GT=img, P=result)[0])

    assert np.mean(similarities) >= 0.9
