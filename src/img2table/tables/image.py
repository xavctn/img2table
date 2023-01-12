# coding: utf-8
import copy
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from img2table.tables.objects.extraction import ExtractedTable
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.cells import get_cells
from img2table.tables.processing.lines import detect_lines
from img2table.tables.processing.tables import get_tables
from img2table.tables.processing.tables.implicit_rows import handle_implicit_rows
from img2table.tables.processing.text.titles import get_title_tables


@dataclass
class TableImage:
    img: np.ndarray
    dpi: int
    ocr_df: "OCRDataframe" = None
    min_confidence: int = 50
    lines: List[Line] = None
    tables: List[Table] = None

    @property
    def white_img(self) -> np.ndarray:
        white_img = copy.deepcopy(self.img)

        # Draw white lines on detected lines
        for line in self.lines:
            cv2.rectangle(white_img, (line.x1, line.y1), (line.x2, line.y2), (255, 255, 255), 3)

        return white_img

    def extract_tables(self, implicit_rows: bool = True) -> List[ExtractedTable]:
        # Detect lines in image
        h_lines, v_lines = detect_lines(image=self.img,
                                        rho=0.3,
                                        theta=np.pi / 180,
                                        threshold=10,
                                        minLinLength=self.dpi // 20,
                                        maxLineGap=self.dpi // 20,
                                        kernel_size=self.dpi // 10,
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
            # Get title
            self.tables = get_title_tables(img=self.img, tables=self.tables, ocr_df=self.ocr_df)

            # Get content
            self.tables = [table.get_content(ocr_df=self.ocr_df, min_confidence=self.min_confidence)
                           for table in self.tables]

        return [table.extracted_table for table in self.tables if table.nb_columns * table.nb_rows > 1]
