# coding: utf-8
import cv2

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.image_segmentation import create_image_segments
from img2table.tables.processing.borderless_tables.model import ImageSegment


def test_create_image_segments():
    img = cv2.imread("test_data/test.bmp", 0)

    result = create_image_segments(img=img,
                                   area=Cell(x1=61, y1=91, x2=390, y2=961),
                                   median_line_sep=16,
                                   char_length=4.66)

    assert result == [ImageSegment(x1=61, y1=704, x2=390, y2=958),
                      ImageSegment(x1=61, y1=662, x2=330, y2=689),
                      ImageSegment(x1=61, y1=93, x2=390, y2=651)]
