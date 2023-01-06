# coding: utf-8
import cv2

from img2table.tables.objects.line import Line
from img2table.tables.processing.lines import overlapping_filter, detect_lines


def test_overlapping_filter():
    # Create lines
    lines = [Line(x1=10, x2=10, y1=10, y2=100),
             Line(x1=11, x2=11, y1=90, y2=120),
             Line(x1=12, x2=12, y1=210, y2=230),
             Line(x1=12, x2=12, y1=235, y2=255),
             Line(x1=20, x2=20, y1=10, y2=100)]

    result = overlapping_filter(lines=lines, max_gap=10)
    expected = [Line(x1=10, x2=10, y1=10, y2=120),
                Line(x1=12, x2=12, y1=210, y2=255),
                Line(x1=20, x2=20, y1=10, y2=100)]

    assert result == expected


def test_detect_lines():
    img = cv2.imread("test_data/test.PNG", cv2.IMREAD_GRAYSCALE)

    result = detect_lines(image=img,
                          rho=0.3,
                          threshold=10,
                          minLinLength=10,
                          maxLineGap=10)
    expected = [
        [Line(x1=157, y1=92, x2=835, y2=92),
         Line(x1=157, y1=168, x2=835, y2=168),
         Line(x1=157, y1=212, x2=835, y2=212),
         Line(x1=157, y1=256, x2=835, y2=256),
         Line(x1=157, y1=299, x2=835, y2=299),
         Line(x1=157, y1=342, x2=835, y2=342),
         Line(x1=157, y1=386, x2=835, y2=386),
         Line(x1=157, y1=430, x2=835, y2=430),
         Line(x1=156, y1=473, x2=836, y2=473)],
        [Line(x1=156, y1=92, x2=156, y2=475),
         Line(x1=434, y1=92, x2=434, y2=474),
         Line(x1=587, y1=92, x2=587, y2=474),
         Line(x1=834, y1=92, x2=834, y2=475)]
    ]

    assert list(result) == expected
