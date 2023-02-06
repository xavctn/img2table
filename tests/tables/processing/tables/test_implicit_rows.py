# coding: utf-8
import json

import cv2
import numpy as np
import polars as pl

from img2table.ocr.data import OCRDataframe
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.tables.implicit_rows import create_word_image, handle_implicit_rows_table, \
    handle_implicit_rows


def test_create_word_image():
    img = cv2.imread("test_data/implicit.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/implicit_ocr_df.csv", sep=";").lazy())

    result = create_word_image(img=img, ocr_df=ocr_df)

    expected = cv2.imread("test_data/word_image.png", cv2.IMREAD_GRAYSCALE)

    assert np.array_equal(result, expected)


def test_handle_implicit_rows_table():
    img = cv2.imread("test_data/word_image.png", cv2.IMREAD_GRAYSCALE)

    with open("test_data/implicit_table.json", 'r') as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    result = handle_implicit_rows_table(img=img, table=table)

    # Check that 2 more lines have been created
    assert result.nb_rows == table.nb_rows + 2


def test_handle_implicit_rows():
    img = cv2.imread("test_data/implicit.png", cv2.IMREAD_GRAYSCALE)
    ocr_df = OCRDataframe(df=pl.read_csv("test_data/implicit_ocr_df.csv", sep=";").lazy())

    with open("test_data/implicit_table.json", 'r') as f:
        table = Table(rows=[Row(cells=[Cell(**el) for el in row]) for row in json.load(f)])

    result = handle_implicit_rows(img=img, tables=[table], ocr_df=ocr_df)

    # Check that 2 more lines have been created
    assert result[0].nb_rows == table.nb_rows + 2
