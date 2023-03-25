# coding: utf-8
import copy
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from img2table.tables.objects.extraction import ExtractedTable
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.cells import get_cells
from img2table.tables.processing.bordered_tables.lines import detect_lines
from img2table.tables.processing.bordered_tables.tables import get_tables
from img2table.tables.processing.bordered_tables.tables.implicit_rows import handle_implicit_rows
from img2table.tables.processing.borderless_tables import identify_borderless_tables
from img2table.tables.processing.prepare_image import prepare_image
from img2table.tables.processing.text.titles import get_title_tables


@dataclass
class TableImage:
    img: np.ndarray
    dpi: int
    ocr_df: "OCRDataframe" = None
    min_confidence: int = 50
    lines: List[Line] = None
    tables: List[Table] = None

    def __post_init__(self):
        # Prepare image by removing eventual black background
        self.img = prepare_image(img=self.img)

    @property
    def white_img(self) -> np.ndarray:
        white_img = copy.deepcopy(self.img)

        # Draw white lines on detected lines
        for line in self.lines:
            cv2.rectangle(white_img, (line.x1, line.y1), (line.x2, line.y2), (255, 255, 255), 3)

        return white_img

    def extract_tables(self, implicit_rows: bool = True, borderless_tables: bool = False) -> List[ExtractedTable]:
        """
        Identify and extract tables from image
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :return: list of identified tables
        """
        # If an OCR is provided, compute parameters base on it
        if self.ocr_df is not None:
            minLinLength = maxLineGap = round(0.33 * self.ocr_df.median_line_sep) if self.ocr_df.median_line_sep is not None else self.dpi // 20
            kernel_size = round(0.66 * self.ocr_df.median_line_sep) if self.ocr_df.median_line_sep is not None else self.dpi // 10
        else:
            minLinLength = maxLineGap = self.dpi // 20
            kernel_size = self.dpi // 10

        # Detect lines in image
        h_lines, v_lines = detect_lines(image=self.img,
                                        rho=0.3,
                                        theta=np.pi / 180,
                                        threshold=10,
                                        minLinLength=minLinLength,
                                        maxLineGap=maxLineGap,
                                        kernel_size=kernel_size,
                                        ocr_df=self.ocr_df)
        self.lines = h_lines + v_lines

        # Create cells from lines
        cells = get_cells(horizontal_lines=h_lines,
                          vertical_lines=v_lines)

        # Create tables from lines
        self.tables = get_tables(cells=cells)

        # If necessary, detect implicit rows
        if implicit_rows:
            self.tables = handle_implicit_rows(img=self.white_img,
                                               tables=self.tables,
                                               ocr_df=self.ocr_df)

        # If ocr_df is available, get titles and tables content
        if self.ocr_df is not None:
            # Get content
            self.tables = [table.get_content(ocr_df=self.ocr_df, min_confidence=self.min_confidence)
                           for table in self.tables]
            self.tables = [table for table in self.tables if table.nb_rows * table.nb_columns > 1]

            if borderless_tables:
                # Extract borderless tables
                borderless_tbs = identify_borderless_tables(img=self.img,
                                                            ocr_df=self.ocr_df,
                                                            lines=self.lines,
                                                            existing_tables=self.tables)

                # Get content
                borderless_tbs = [table.get_content(ocr_df=self.ocr_df, min_confidence=self.min_confidence)
                                  for table in borderless_tbs]

                # Add to tables
                self.tables += [tb for tb in borderless_tbs if min(tb.nb_rows, tb.nb_columns) >= 2]

            # Get title
            self.tables = get_title_tables(img=self.img, tables=self.tables, ocr_df=self.ocr_df)

        return [table.extracted_table for table in self.tables]
