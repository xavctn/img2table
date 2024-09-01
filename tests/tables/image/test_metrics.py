# coding: utf-8
import cv2

from img2table.tables import threshold_dark_areas
from img2table.tables.metrics import compute_char_length, compute_median_line_sep, compute_img_metrics


def test_compute_char_length():
    image = cv2.cvtColor(cv2.imread("test_data/test.png"), cv2.COLOR_BGR2RGB)
    thresh = threshold_dark_areas(img=image, char_length=11)

    char_length, thresh_chars, chars_array = compute_char_length(thresh=thresh)
    assert round(char_length, 2) == 9.0
    assert thresh_chars.shape == (417, 1365)

    image = 255 - cv2.cvtColor(cv2.imread("test_data/blank.png"), cv2.COLOR_BGR2GRAY)
    assert compute_char_length(thresh=image) == (None, None, None)


def test_compute_median_line_sep():
    image = cv2.cvtColor(cv2.imread("test_data/test.png"), cv2.COLOR_BGR2RGB)
    thresh = threshold_dark_areas(img=image, char_length=11)
    char_length, thresh_chars, chars_array = compute_char_length(thresh=thresh)

    median_line_sep, contours = compute_median_line_sep(thresh_chars=thresh_chars,
                                                        chars_array=chars_array,
                                                        char_length=char_length)

    assert round(median_line_sep, 2) == 51
    assert len(contours) == 71


def test_compute_img_metrics():
    image = cv2.cvtColor(cv2.imread("test_data/test.png"), cv2.COLOR_BGR2RGB)
    thresh = threshold_dark_areas(img=image, char_length=11)
    char_length, median_line_sep, contours = compute_img_metrics(thresh=thresh)

    assert round(char_length, 2) == 9.0
    assert round(median_line_sep, 2) == 51
    assert len(contours) == 71

    image = 255 - cv2.cvtColor(cv2.imread("test_data/blank.png"), cv2.COLOR_BGR2GRAY)
    assert compute_img_metrics(thresh=image) == (None, None, None)
