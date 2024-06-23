# coding: utf-8
import cv2

from img2table.tables.metrics import compute_char_length, compute_median_line_sep, compute_img_metrics


def test_compute_char_length():
    image = cv2.cvtColor(cv2.imread("test_data/test.png"), cv2.COLOR_BGR2RGB)

    char_length, thresh_chars = compute_char_length(img=image)
    assert round(char_length, 2) == 9.0
    assert thresh_chars.shape == (417, 1365)

    image = cv2.cvtColor(cv2.imread("test_data/blank.png"), cv2.COLOR_BGR2RGB)
    assert compute_char_length(img=image) == (None, None)


def test_compute_median_line_sep():
    image = cv2.cvtColor(cv2.imread("test_data/test.png"), cv2.COLOR_BGR2RGB)
    char_length, thresh_chars = compute_char_length(img=image)

    median_line_sep, contours = compute_median_line_sep(thresh_chars=thresh_chars,
                                                        char_length=char_length)

    assert round(median_line_sep, 2) == 51
    assert len(contours) == 71


def test_compute_img_metrics():
    image = cv2.cvtColor(cv2.imread("test_data/test.png"), cv2.COLOR_BGR2RGB)
    char_length, median_line_sep, contours = compute_img_metrics(img=image)

    assert round(char_length, 2) == 9.0
    assert round(median_line_sep, 2) == 51
    assert len(contours) == 71

    image = cv2.cvtColor(cv2.imread("test_data/blank.png"), cv2.COLOR_BGR2RGB)
    assert compute_img_metrics(img=image) == (None, None, None)
