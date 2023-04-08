# coding: utf-8
import json

import cv2
import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.line import Line
from img2table.tables.processing.borderless_tables.model import ImageSegment
from img2table.tables.processing.borderless_tables.segment_image import create_image_segments, get_segment_elements, \
    segment_image


def test_create_image_segments():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";").lazy())

    result = create_image_segments(img=img, ocr_df=ocr_df)

    assert result == [ImageSegment(x1=2, y1=0, x2=804, y2=361), ImageSegment(x1=928, y1=0, x2=1188, y2=157)]


def test_get_segment_elements():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";").lazy())
    img_segments = [ImageSegment(x1=2, y1=0, x2=804, y2=361),
                    ImageSegment(x1=928, y1=0, x2=1188, y2=157)]

    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = get_segment_elements(img=img,
                                  lines=lines,
                                  img_segments=img_segments,
                                  blur_size=3,
                                  ocr_df=ocr_df)

    assert len(result[0].elements) == 14
    assert len(result[1].elements) == 4


def test_segment_image():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", separator=";").lazy())
    with open("test_data/lines.json", 'r') as f:
        data = json.load(f)
    lines = [Line(**el) for el in data.get('h_lines') + data.get('v_lines')]

    result = segment_image(img=img,
                           ocr_df=ocr_df,
                           lines=lines)

    assert len(result[0].elements) == 14
    assert len(result[1].elements) == 4
