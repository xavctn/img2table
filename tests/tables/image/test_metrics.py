# coding: utf-8
import cv2

from img2table.tables.metrics import compute_char_length, compute_median_line_sep, compute_img_metrics


def test_compute_char_length():
    image = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    char_length, cc_array = compute_char_length(img=image)
    assert round(char_length, 2) == 8.44
    assert len(cc_array) == 117

    image = cv2.imread("test_data/blank.png", cv2.IMREAD_GRAYSCALE)
    assert compute_char_length(img=image) == (None, None)


def test_compute_median_line_sep():
    image = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    char_length, cc_array = compute_char_length(img=image)

    median_line_sep, contours = compute_median_line_sep(img=image, char_length=char_length, cc=cc_array)

    assert round(median_line_sep, 2) == 51
    assert len(contours) == 43


def test_compute_img_metrics():
    image = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    char_length, median_line_sep, contours = compute_img_metrics(img=image)

    assert round(char_length, 2) == 8.44
    assert round(median_line_sep, 2) == 51
    assert len(contours) == 43

    image = cv2.imread("test_data/blank.png", cv2.IMREAD_GRAYSCALE)
    assert compute_img_metrics(img=image) == (None, None, None)
