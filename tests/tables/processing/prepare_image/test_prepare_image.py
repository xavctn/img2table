# coding: utf-8
import cv2
import numpy as np

from img2table.tables.processing.prepare_image import prepare_image


def test_prepare_image():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    result = prepare_image(img=img)

    # Check that the average value of a pixel is darker in original image
    assert np.mean(img) < np.mean(result)
