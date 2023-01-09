# coding: utf-8
import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.common import is_contained_cell, merge_contours, get_contours_cell


def test_is_contained_cell():
    cell_1 = Cell(x1=0, x2=20, y1=0, y2=20)
    cell_2 = Cell(x1=0, x2=40, y1=0, y2=25)
    cell_3 = Cell(x1=50, x2=70, y1=123, y2=256)

    assert is_contained_cell(inner_cell=cell_1, outer_cell=cell_2)
    assert not is_contained_cell(inner_cell=cell_2, outer_cell=cell_1)
    assert not is_contained_cell(inner_cell=cell_1, outer_cell=cell_3)
    assert not is_contained_cell(inner_cell=cell_2, outer_cell=cell_3)


def test_merge_contours():
    contours = [Cell(x1=0, x2=20, y1=0, y2=20),
                Cell(x1=0, x2=20, y1=10, y2=20),
                Cell(x1=60, x2=80, y1=0, y2=20),
                Cell(x1=10, x2=20, y1=100, y2=200)]

    # Do not merge by axis
    expected = [Cell(x1=0, x2=20, y1=0, y2=20),
                Cell(x1=60, x2=80, y1=0, y2=20),
                Cell(x1=10, x2=20, y1=100, y2=200)]
    assert set(merge_contours(contours=contours, vertically=None)) == set(expected)

    # Merge vertically
    expected_vertical = [Cell(x1=0, x2=80, y1=0, y2=20), Cell(x1=10, x2=20, y1=100, y2=200)]
    assert merge_contours(contours=contours, vertically=True) == expected_vertical

    # Merge horizontally
    expected_horizontal = [Cell(x1=0, x2=20, y1=0, y2=200), Cell(x1=60, x2=80, y1=0, y2=20)]
    assert merge_contours(contours=contours, vertically=False) == expected_horizontal


def test_get_contours_cell():
    img = cv2.imread("test_data/test.jpg", cv2.IMREAD_GRAYSCALE)
    cell = Cell(x1=0, x2=img.shape[1], y1=0, y2=img.shape[0])

    result = get_contours_cell(img=img,
                               cell=cell,
                               margin=5,
                               blur_size=5,
                               kernel_size=9,
                               merge_vertically=True)

    expected = [Cell(x1=51, y1=19, x2=518, y2=146),
                Cell(x1=60, y1=156, x2=534, y2=691),
                Cell(x1=52, y1=765, x2=543, y2=811)]

    assert result == expected
