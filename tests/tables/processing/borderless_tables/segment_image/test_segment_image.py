# coding: utf-8
import json

import cv2
import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.segment_image import create_image_segments, \
    text_countours_from_group, create_word_contours, segment_image_text


def test_create_image_segments():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    result = create_image_segments(img=img, ocr_df=ocr_df)

    assert set(result) == {Cell(x1=2, y1=0, x2=804, y2=361), Cell(x1=928, y1=0, x2=1188, y2=157)}


def test_text_countours_from_group():
    word_group = [{"x1": 0, "x2": 10, "y1": 0, "y2": 10, "length": 3},
                  {"x1": 13, "x2": 26, "y1": 0, "y2": 10, "length": 5},
                  {"x1": 78, "x2": 87, "y1": 0, "y2": 10, "length": 2},
                  {"x1": 0, "x2": 10, "y1": 100, "y2": 110, "length": 3}
                  ]

    result = text_countours_from_group(word_group=word_group)

    assert result == [Cell(x1=0, y1=0, x2=26, y2=10),
                      Cell(x1=78, y1=0, x2=87, y2=10),
                      Cell(x1=0, y1=100, x2=10, y2=110)]


def test_create_word_contours():
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    result = create_word_contours(ocr_df=ocr_df)

    with open("test_data/word_contours.json", "r") as f:
        expected = [Cell(**element) for element in json.load(f)]

    assert set(result) == set(expected)


def test_segment_image_text():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/ocr_df.csv", sep=";").lazy())

    result = segment_image_text(img=img, ocr_df=ocr_df)

    with open("test_data/expected.json", "r") as f:
        expected = [[Cell(**element) for element in seg] for seg in json.load(f)]

    assert [set(seg) for seg in result] == [set(seg) for seg in expected]
