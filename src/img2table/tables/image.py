import copy
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from img2table.tables import threshold_dark_areas
from img2table.tables.metrics import compute_img_metrics
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.cells import get_cells
from img2table.tables.processing.bordered_tables.lines import detect_lines
from img2table.tables.processing.bordered_tables.tables import get_tables
from img2table.tables.processing.bordered_tables.tables.consecutive import merge_consecutive_tables
from img2table.tables.processing.bordered_tables.tables.implicit import implicit_content
from img2table.tables.processing.borderless_tables import identify_borderless_tables


@dataclass
class TableImage:
    img: np.ndarray
    min_confidence: int = 50
    char_length: float = None
    median_line_sep: float = None
    thresh: np.ndarray = None
    contours: list[Cell] = None
    lines: list[Line] = None
    tables: list[Table] = None

    def __post_init__(self) -> None:
        self.thresh = threshold_dark_areas(img=self.img, char_length=11)

        # Compute image metrics
        self.char_length, self.median_line_sep, self.contours = compute_img_metrics(thresh=self.thresh.copy())

    @cached_property
    def white_img(self) -> np.ndarray:
        white_img = copy.deepcopy(self.img)

        # Draw white rows on detected rows
        for line in self.lines:
            if line.horizontal:
                cv2.rectangle(white_img, (line.x1 - line.thickness, line.y1), (line.x2 + line.thickness, line.y2),
                              (255, 255, 255), 3 * line.thickness)
            elif line.vertical:
                cv2.rectangle(white_img, (line.x1, line.y1 - line.thickness), (line.x2, line.y2 + line.thickness),
                              (255, 255, 255), 2 * line.thickness)

        return white_img

    def extract_bordered_tables(self, implicit_rows: bool = False, implicit_columns: bool = False) -> None:
        """
        Identify and extract bordered tables from image
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :return:
        """
        # Compute parameters for line detection
        min_line_length = int(min(1.5 * self.median_line_sep, 4 * self.char_length)) if self.median_line_sep else 20

        # Detect rows in image
        h_lines, v_lines = detect_lines(img=self.img,
                                        contours=self.contours,
                                        char_length=self.char_length,
                                        min_line_length=min_line_length)
        self.lines = h_lines + v_lines

        # Create cells from rows
        cells = get_cells(horizontal_lines=h_lines,
                          vertical_lines=v_lines)

        # Create tables from rows
        self.tables = get_tables(cells=cells,
                                 elements=self.contours,
                                 lines=self.lines,
                                 char_length=self.char_length)

        # If necessary, detect implicit rows
        self.tables = [implicit_content(table=table,
                                        contours=self.contours,
                                        char_length=self.char_length,
                                        implicit_rows=implicit_rows,
                                        implicit_columns=implicit_columns)
                       for table in self.tables]

        # Merge consecutive tables
        self.tables = merge_consecutive_tables(tables=self.tables,
                                               contours=self.contours)

        # Post filter bordered tables
        self.tables = [tb for tb in self.tables if min(tb.nb_rows, tb.nb_columns) >= 2]

    def extract_borderless_tables(self) -> None:
        """
        Identify and extract borderless tables from image
        :return:
        """
        # Median line separation needs to be not null to extract borderless tables
        if self.median_line_sep is not None:
            self.thresh = threshold_dark_areas(img=self.img, char_length=self.char_length)

            # Extract borderless tables
            borderless_tbs = identify_borderless_tables(thresh=self.thresh,
                                                        char_length=self.char_length,
                                                        median_line_sep=self.median_line_sep,
                                                        lines=self.lines,
                                                        contours=self.contours,
                                                        existing_tables=self.tables)

            # Add to tables
            self.tables += [tb for tb in borderless_tbs if tb.nb_rows >= 2 and tb.nb_columns >= 3]

    def extract_tables(self, implicit_rows: bool = False, implicit_columns: bool = False, borderless_tables: bool = False) -> list[Table]:
        """
        Identify and extract tables from image
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :return: list of identified tables
        """
        if self.char_length is None:
            return []

        # Extract bordered tables
        self.extract_bordered_tables(implicit_rows=implicit_rows,
                                     implicit_columns=implicit_columns)

        if borderless_tables:
            # Extract borderless tables
            self.extract_borderless_tables()

        return self.tables
