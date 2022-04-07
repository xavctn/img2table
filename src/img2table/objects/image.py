# coding: utf-8
import copy
from typing import List

import numpy as np
from cv2 import cv2

from img2table.objects.ocr import OCRPage
from img2table.objects.tables import Table, Line, Cell
from img2table.utils.cell_detection import get_cells
from img2table.utils.implicit_rows import handle_implicit_rows
from img2table.utils.implicit_tables import detect_implicit_tables
from img2table.utils.line_detection import detect_lines
from img2table.utils.rotation import rotate_img
from img2table.utils.table_detection import get_tables
from img2table.utils.text_extraction import get_text_tables


class TableImage(object):
    def __init__(self, image: np.ndarray, lang: str = "fra+eng"):
        self._original_img = image
        self._lang = lang
        # Rotate image to proper orientation
        self._img = rotate_img(img=self.original_img)

        # Initialize attributes
        self._ocr_page = None
        self._h_lines = []
        self._v_lines = []
        self._tables = []
        self._implicit_tables = []
        self._white_img = None

    @property
    def original_img(self) -> np.ndarray:
        return copy.deepcopy(self._original_img)

    @property
    def img(self) -> np.ndarray:
        return copy.deepcopy(self._img)

    @property
    def white_img(self) -> np.ndarray:
        return copy.deepcopy(self._white_img)

    @property
    def lang(self) -> str:
        return self._lang

    @property
    def ocr_page(self) -> OCRPage:
        return self._ocr_page

    @property
    def tables(self) -> List[Table]:
        return [table for table in self._tables if table is not None]

    @property
    def implicit_tables(self) -> List[Table]:
        return [table for table in self._implicit_tables if table is not None]

    @property
    def total_tables(self) -> List[Table]:
        return self.tables + self.implicit_tables

    @property
    def h_lines(self) -> List[Line]:
        return copy.deepcopy(self._h_lines)

    @property
    def v_lines(self) -> List[Line]:
        return copy.deepcopy(self._v_lines)

    def _identify_lines(self, rho: float = 0.3, theta: float = np.pi / 180, threshold: int = 10,
                        minLinLength: int = 10, maxLineGap: int = 10):
        """
        Identify horizontal lines in image
        :param rho: rho parameter for Hough line transform
        :param theta: theta parameter for Hough line transform
        :param threshold: threshold parameter for Hough line transform
        :param minLinLength: minLinLength parameter for Hough line transform
        :param maxLineGap: maxLineGap parameter for Hough line transform
        """
        # Set _h_lines and _v_lines attributes
        self._h_lines, self._v_lines = detect_lines(image=self.img,
                                                    rho=rho,
                                                    theta=theta,
                                                    threshold=threshold,
                                                    minLinLength=minLinLength,
                                                    maxLineGap=maxLineGap)

    def _identify_cells(self) -> List[Cell]:
        return get_cells(horizontal_lines=self.h_lines,
                         vertical_lines=self.v_lines)

    def _detect_tables_from_lines(self):
        """
        Identify tables using vertical and horizontal lines in image
        :return:
        """
        # Identify horizontal and vertical lines
        self._identify_lines()

        # Identify cells
        cells = self._identify_cells()

        # Get tables from horizontal and vertical lines
        self._tables = get_tables(cells=cells)

    def _create_img_colored_borders(self, color: tuple = (255, 255, 255), margin: int = 1):
        """
        Draw white lines on cell borders in order to improve OCR performance
        :param color: RGB color code
        :param margin: margin used around cell borders to draw lines
        :return:
        """
        # Initialize image
        self._white_img = self.img

        # Draw white lines on cells borders
        for cell in [cell for table in self.total_tables for row in table.items for cell in row.items]:
            cv2.rectangle(self._white_img, (cell.x1, cell.y1 - margin), (cell.x2, cell.y1 + margin), color, 3)
            cv2.rectangle(self._white_img, (cell.x1, cell.y2 - margin), (cell.x2, cell.y2 + margin), color, 3)
            cv2.rectangle(self._white_img, (cell.x1 - margin, cell.y1), (cell.x1 + margin, cell.y2), color, 3)
            cv2.rectangle(self._white_img, (cell.x2 - margin, cell.y1), (cell.x2 + margin, cell.y2), color, 3)

    def _white_lines(self, color: tuple = (255, 255, 255)) -> np.ndarray:
        """
        Draw white lines on identified lines
        :param color: RGB color code
        :return: image
        """
        # Initialize image
        _img = self.img

        # Draw white lines on cells borders
        for line in self.h_lines + self.v_lines:
            cv2.rectangle(_img, (line.x1, line.y1), (line.x2, line.y2), color, 3)

        return _img

    def _detect_implicit_rows(self):
        """
        Detect implicit rows in tables
        :return:
        """
        self._tables = handle_implicit_rows(white_img=self.white_img,
                                            tables=self.tables)

    def identify_image_tables(self, implicit_rows: bool = True, implicit_tables: bool = True) -> List[Table]:
        """
        Identify tables in image
        :param implicit_rows: boolean indicating if implicit rows are detected
        :param implicit_tables: boolean indicating if implicit tables are detected
        :return: list of Table objects
        """
        # Detect tables from lines
        self._detect_tables_from_lines()

        # Color table in white
        self._create_img_colored_borders(margin=0)

        if implicit_rows:
            self._detect_implicit_rows()

        if implicit_tables:
            self._implicit_tables = detect_implicit_tables(white_img=self._white_lines(),
                                                           tables=self.tables)

        return self.total_tables

    def parse_tables_content(self) -> List[Table]:
        """
        Parse table to pandas DataFrames and set titles
        :return: list of parsed Table objects
        """
        # Remove tables that have only one cell
        self._tables = [table for table in self.tables if table.nb_rows * table.nb_columns > 1]

        if self.total_tables:
            # Create OCRPage object
            self._ocr_page = OCRPage.of(image=self.white_img, lang=self.lang)

            self._tables = get_text_tables(img=self.white_img,
                                           ocr_page=self.ocr_page,
                                           tables=self.tables)

            self._implicit_tables = get_text_tables(img=self.white_img,
                                                    ocr_page=self.ocr_page,
                                                    tables=self._implicit_tables)

        return self.total_tables

    def extract_tables(self, implicit_rows: bool = True, implicit_tables: bool = False) -> List[Table]:
        """
        Extract tables from image
        :param implicit_rows: boolean indicating if implicit rows are detected
        :param implicit_tables: boolean indicating if implicit tables are detected
        :return: list of extracted tables
        """
        self.identify_image_tables(implicit_rows=implicit_rows,
                                   implicit_tables=implicit_tables)

        extracted_tables = self.parse_tables_content()

        return extracted_tables


if __name__ == '__main__':
    from PIL import Image as PILImage

    img = cv2.imread(r"C:\Users\xavca\Pictures\data_3.jpeg")

    image_object = TableImage(img)
    tables = image_object.extract_tables(implicit_rows=True,
                                         implicit_tables=True)

    image_object._create_img_colored_borders(color=(128, 145, 226), margin=0)
    display_img = image_object.white_img

    PILImage.fromarray(display_img).show()

    output_tables = [{"title": table.title,
                      "bbox": table.bbox(),
                      "data": table.data}
                     for table in tables]

    for table in output_tables:
        print(table.get('title'))
        print(table.get('bbox'))
        print(table.get('data').head(20))
