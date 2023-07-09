# coding: utf-8
import json

import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.image_segmentation.column_segmentation import same_column, \
    coherent_columns, intertwined_col_groups, ColumnGroup, get_image_columns, identify_remaining_segments, \
    segment_image_columns


def test_same_column():
    col_1 = Cell(x1=0, y1=0, x2=20, y2=20)
    col_2 = Cell(x1=0, y1=30, x2=20, y2=50)
    col_3 = Cell(x1=10, y1=30, x2=30, y2=50)

    assert same_column(median_line_sep=5, cnt_1=col_1, cnt_2=col_2)
    assert not same_column(median_line_sep=3, cnt_1=col_1, cnt_2=col_2)
    assert not same_column(median_line_sep=5, cnt_1=col_1, cnt_2=col_3)


def test_coherent_columns():
    col_1 = Cell(x1=0, y1=0, x2=20, y2=20)
    col_2 = Cell(x1=100, y1=2, x2=120, y2=50)
    col_3 = Cell(x1=100, y1=2, x2=110, y2=50)
    col_4 = Cell(x1=100, y1=12, x2=120, y2=50)

    assert coherent_columns(col_1=col_1, col_2=col_2)
    assert not coherent_columns(col_1=col_1, col_2=col_3)
    assert not coherent_columns(col_1=col_1, col_2=col_4)


def test_intertwined_col_groups():
    columns = [Cell(x1=0, y1=0, x2=20, y2=100),
               Cell(x1=30, y1=0, x2=50, y2=100),
               Cell(x1=60, y1=0, x2=80, y2=100),
               Cell(x1=100, y1=0, x2=110, y2=100)]
    col_gp_1 = ColumnGroup(columns=[Cell(x1=0, y1=0, x2=20, y2=100),
                                    Cell(x1=30, y1=0, x2=50, y2=100),
                                    Cell(x1=60, y1=0, x2=80, y2=100)])
    col_gp_2 = ColumnGroup(columns=[Cell(x1=0, y1=0, x2=20, y2=100),
                                    Cell(x1=30, y1=0, x2=50, y2=100),
                                    Cell(x1=100, y1=0, x2=110, y2=100)])

    assert not intertwined_col_groups(col_gp=col_gp_1, columns=columns)
    assert intertwined_col_groups(col_gp=col_gp_2, columns=columns)


def test_get_image_columns():
    img = cv2.imread("test_data/test.bmp", 0)

    result = get_image_columns(img=img,
                               median_line_sep=16,
                               char_length=4.66)

    assert result == [ColumnGroup(columns=[Cell(x1=392, y1=100, x2=750, y2=960),
                                           Cell(x1=61, y1=91, x2=389, y2=961)])
                      ]


def test_identify_remaining_segments():
    existing_segments = [Cell(x1=61, y1=91, x2=750, y2=961)]

    result = identify_remaining_segments(existing_segments=existing_segments,
                                         height=1056,
                                         width=816)

    assert result == [Cell(x1=0, y1=961, x2=816, y2=1056),
                      Cell(x1=0, y1=0, x2=816, y2=91),
                      Cell(x1=750, y1=91, x2=816, y2=961),
                      Cell(x1=0, y1=91, x2=61, y2=961)]


def test_segment_image_columns():
    img = cv2.imread("test_data/test.bmp", 0)

    with open("test_data/contours.json", 'r') as f:
        contours = [Cell(**el) for el in json.load(f)]

    result = segment_image_columns(img=img,
                                   median_line_sep=16,
                                   char_length=4.66,
                                   contours=contours)

    assert result == [Cell(x1=61, y1=91, x2=390, y2=961),
                      Cell(x1=390, y1=91, x2=750, y2=961)]
