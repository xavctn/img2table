# coding: utf-8

import cv2

from img2table.tables.image import TableImage


def test_table_image():
    image = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    tb_image = TableImage(img=image,
                          min_confidence=50)

    result = tb_image.extract_tables(implicit_rows=True)
    result = sorted(result, key=lambda tb: tb.x1 + tb.x2)

    assert (result[0].x1, result[0].y1, result[0].x2, result[0].y2) == (36, 21, 770, 327)
    assert (result[0].nb_rows, result[0].nb_columns) == (6, 3)

    assert (result[1].x1, result[1].y1, result[1].x2, result[1].y2) == (962, 21, 1154, 123)
    assert (result[1].nb_rows, result[1].nb_columns) == (2, 2)
