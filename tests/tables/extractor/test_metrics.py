import cv2

from img2table.tables.common import threshold_dark_areas
from img2table.tables.metrics import (
    compute_char_length,
    compute_img_metrics,
    compute_median_line_sep,
)


def test_compute_char_length() -> None:
    img = cv2.imread("test_data/test.png")
    assert img is not None
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    thresh = threshold_dark_areas(img=img, char_length=11)

    char_length, chars_array = compute_char_length(thresh=thresh)
    assert char_length is not None
    assert round(char_length, 2) == 9.0
    assert chars_array is not None
    assert chars_array.shape[0] == 171

    img = cv2.imread("test_data/blank.png")
    assert img is not None
    img = 255 - cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    assert compute_char_length(thresh=img) == (None, None)


def test_compute_median_line_sep() -> None:
    img = cv2.imread("test_data/test.png")
    assert img is not None
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    thresh = threshold_dark_areas(img=img, char_length=11)
    char_length, chars_array = compute_char_length(thresh=thresh)

    assert chars_array is not None
    assert char_length is not None

    median_line_sep, contours = compute_median_line_sep(
        chars_array=chars_array,
        char_length=char_length,
        height=img.shape[0],
        width=img.shape[1],
    )

    assert median_line_sep is not None
    assert round(median_line_sep, 2) == 51
    assert contours is not None
    assert len(contours) == 86


def test_compute_img_metrics() -> None:
    img = cv2.imread("test_data/test.png")
    assert img is not None
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    thresh = threshold_dark_areas(img=img, char_length=11)
    char_length, median_line_sep, contours = compute_img_metrics(thresh=thresh)

    assert char_length is not None
    assert round(char_length, 2) == 9.0
    assert median_line_sep is not None
    assert round(median_line_sep, 2) == 51
    assert contours is not None
    assert len(contours) == 86

    img = cv2.imread("test_data/blank.png")
    assert img is not None
    img = 255 - cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    assert compute_img_metrics(thresh=img) == (None, None, None)
