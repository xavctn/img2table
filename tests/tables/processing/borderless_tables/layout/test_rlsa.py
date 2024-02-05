# coding: utf-8
import json

import cv2
import numpy as np
from numba import config

from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.layout.rlsa import identify_text_mask


def test_identify_text_mask():
    config.DISABLE_JIT = True

    img = cv2.imread("test_data/test.bmp", cv2.IMREAD_GRAYSCALE)

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = identify_text_mask(img=img,
                                lines=lines,
                                char_length=6.0)

    expected = cv2.imread("test_data/text_thresh.bmp", cv2.IMREAD_GRAYSCALE)

    assert np.array_equal(result, expected)
