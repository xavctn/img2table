# coding: utf-8
import json

import cv2

from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.image_segmentation import get_segment_elements
from img2table.tables.processing.borderless_tables.model import ImageSegment


def test_get_segment_elements():
    img = cv2.imread("test_data/test.bmp", cv2.IMREAD_GRAYSCALE)
    img_segments = [ImageSegment(x1=61, y1=92, x2=390, y2=652)]

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = get_segment_elements(img=img,
                                  lines=lines,
                                  img_segments=img_segments,
                                  blur_size=3,
                                  char_length=4.66,
                                  median_line_sep=16)

    assert len(result[0].elements) == 67
