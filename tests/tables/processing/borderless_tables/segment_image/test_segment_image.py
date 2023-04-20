# coding: utf-8
import json

import cv2

from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.model import ImageSegment
from img2table.tables.processing.borderless_tables.segment_image import create_image_segments, get_segment_elements, \
    segment_image


def test_create_image_segments():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    result = create_image_segments(img=img, median_line_sep=51)

    assert set(result) == {ImageSegment(x1=2, y1=0, x2=804, y2=361), ImageSegment(x1=928, y1=0, x2=1188, y2=157)}


def test_get_segment_elements():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    img_segments = [ImageSegment(x1=2, y1=0, x2=804, y2=361),
                    ImageSegment(x1=928, y1=0, x2=1188, y2=157)]

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = get_segment_elements(img=img,
                                  lines=lines,
                                  img_segments=img_segments,
                                  blur_size=3,
                                  char_length=8.44,
                                  median_line_sep=51)

    assert len(result[0].elements) == 14
    assert len(result[1].elements) == 4


def test_segment_image():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = segment_image(img=img,
                           char_length=8.44,
                           median_line_sep=51,
                           lines=lines)

    assert {len(result[0].elements), len(result[1].elements)} == {14, 4}
