# coding: utf-8
import copy
from typing import List

import numpy as np
from cv2 import cv2

from img2table.objects.tables import Table, Line
from img2table.utils.column_detection import get_columns_table
from img2table.utils.implicit_rows import handle_implicit_rows
from img2table.utils.line_detection import detect_lines
from img2table.utils.rotation import rotate_img
from img2table.utils.table_detection import get_tables
from img2table.utils.text_extraction import get_text_tables


class Image(object):
    def __init__(self, image: np.ndarray):
        self._original_img = image
        # Rotate image to proper orientation
        self._img = rotate_img(img=copy.deepcopy(image))

        # Initialize attributes
        self._h_lines = []
        self._v_lines = []
        self._tables = []
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
    def tables(self) -> List[Table]:
        return [table for table in self._tables if table is not None]

    @property
    def h_lines(self) -> List[Line]:
        return copy.deepcopy(self._h_lines)

    @property
    def v_lines(self) -> List[Line]:
        return copy.deepcopy(self._v_lines)

    def _identify_horizontal_lines(self, rho: float = 0.5, theta: float = np.pi / 180, threshold: int = 50,
                                   minLinLength: int = 200, maxLineGap: int = 6):
        """
        Identify horizontal lines in image
        :param rho: rho parameter for Hough line transform
        :param theta: theta parameter for Hough line transform
        :param threshold: threshold parameter for Hough line transform
        :param minLinLength: minLinLength parameter for Hough line transform
        :param maxLineGap: maxLineGap parameter for Hough line transform
        """
        horizontal_lines, _ = detect_lines(image=self.img,
                                           rho=rho,
                                           theta=theta,
                                           threshold=threshold,
                                           minLinLength=minLinLength,
                                           maxLineGap=maxLineGap)

        # Set _h_lines attribute
        self._h_lines = horizontal_lines

    def _identify_vertical_lines(self, rho: float = 0.5, theta: float = np.pi / 180, threshold: int = 5,
                                 minLinLength: int = 20, maxLineGap: int = 1):
        """
        Identify vertical lines in image
        :param rho: rho parameter for Hough line transform
        :param theta: theta parameter for Hough line transform
        :param threshold: threshold parameter for Hough line transform
        :param minLinLength: minLinLength parameter for Hough line transform
        :param maxLineGap: maxLineGap parameter for Hough line transform
        """
        _, vertical_lines = detect_lines(image=self.img,
                                         rho=rho,
                                         theta=theta,
                                         threshold=threshold,
                                         minLinLength=minLinLength,
                                         maxLineGap=maxLineGap)

        # Set _h_lines attribute
        self._v_lines = vertical_lines

    def _detect_tables_from_lines(self):
        """
        Identify tables using vertical and horizontal lines in image
        :return:
        """
        # Identify horizontal and vertical lines
        self._identify_horizontal_lines()
        self._identify_vertical_lines()

        # Get tables from horizontal and vertical lines
        self._tables = get_tables(horizontal_lines=self._h_lines,
                                  vertical_lines=self.v_lines)

    def _create_img_colored_borders(self, color: tuple = (255, 255, 255), margin: int = 1):
        """
        Draw white lines on cell borders in order to improve OCR performance
        :param color: RGB color code
        :param margin: margin used around cell borders to draw lines
        :return:
        """
        # Initialize image
        white_img = self.img

        # Draw white lines on cells borders
        for cell in [cell for table in self.tables for row in table.items for cell in row.items]:
            cv2.rectangle(white_img, (cell.x1, cell.y1 - margin), (cell.x2, cell.y1 + margin), color, 3)
            cv2.rectangle(white_img, (cell.x1, cell.y2 - margin), (cell.x2, cell.y2 + margin), color, 3)
            cv2.rectangle(white_img, (cell.x1 - margin, cell.y1), (cell.x1 + margin, cell.y2), color, 3)
            cv2.rectangle(white_img, (cell.x2 - margin, cell.y1), (cell.x2 + margin, cell.y2), color, 3)

        # Set _white_img attribute
        self._white_img = white_img

    def _detect_columns(self):
        """
        Detect columns in image tables
        :return:
        """
        tables_with_columns = [get_columns_table(img=self.img,
                                                 table=table,
                                                 vertical_lines=self.v_lines)
                               for table in self._tables]

        # Set _tables attribute
        self._tables = tables_with_columns

        # Create image with white table borders
        self._create_img_colored_borders(margin=1)

    def _detect_implicit_rows(self):
        """
        Detect implicit rows in tables
        :return:
        """
        self._tables = handle_implicit_rows(white_img=self.white_img,
                                            tables=self.tables)

    def identify_image_tables(self, implicit_rows: bool = True) -> List[Table]:
        """
        Identify tables in image
        :param implicit_rows: boolean indicating if implicit rows are detected
        :return: list of Table objects
        """
        # Detect tables from lines
        self._detect_tables_from_lines()

        # Identify columns in tables
        self._detect_columns()

        if implicit_rows:
            self._detect_implicit_rows()

        return self.tables

    def parse_tables_content(self, header_detection: bool = True) -> List[Table]:
        """
        Parse table to pandas DataFrames and set titles
        :param header_detection: boolean indicating if header detection is performed
        :return: list of parsed Table objects
        """
        self._tables = get_text_tables(img=self.white_img,
                                       tables=self.tables,
                                       header_detection=header_detection)

        return self.tables

    def extract_tables(self, implicit_rows: bool = True, header_detection: bool = True) -> List[Table]:
        """
        Extract tables from image
        :param implicit_rows: boolean indicating if implicit rows are detected
        :param header_detection: boolean indicating if header detection is performed
        :return: list of extracted tables
        """
        self.identify_image_tables(implicit_rows=implicit_rows)

        extracted_tables = self.parse_tables_content(header_detection=header_detection)

        return extracted_tables


if __name__ == '__main__':
    from PIL import Image as PILImage
    img = cv2.imread(r"C:\Users\xavca\Pictures\achat_blouson.png")

    image_object = Image(img)
    tables = image_object.extract_tables(header_detection=True, implicit_rows=True)
    image_object._create_img_colored_borders(color=(128, 145, 226))

    PILImage.fromarray(image_object.white_img).convert('RGB').show()

    output_tables = [{"title": table.title,
                      "bbox": table.bbox(),
                      "data": table.data}
                     for table in tables]

    for table in output_tables:
        print(table.get('title'))
        print(table.get('bbox'))

