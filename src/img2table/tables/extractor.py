from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from img2table.tables.bordered import extract_bordered_tables
from img2table.tables.bordered.lines import detect_lines
from img2table.tables.borderless import extract_borderless_tables
from img2table.tables.common import get_title_areas, threshold_dark_areas
from img2table.tables.metrics import compute_img_metrics

if TYPE_CHECKING:
    import numpy as np

    from img2table.tables.types import Cell, Line, Table


@dataclass
class ImgCharacteristics:
    char_length: float
    median_line_sep: float
    contours: list[Cell]

    @property
    def min_line_length(self) -> int:
        return int(min(1.5 * self.median_line_sep, 4 * self.char_length))


@dataclass
class TableExtractor:
    img: np.ndarray
    characteristics: ImgCharacteristics | None = None
    lines: list[Line] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.thresh = threshold_dark_areas(img=self.img, char_length=11)

        # Compute image metrics
        char_length, median_line_sep, contours = compute_img_metrics(thresh=self.thresh)
        if char_length is not None and median_line_sep is not None and contours is not None:
            self.characteristics = ImgCharacteristics(
                char_length=char_length, median_line_sep=median_line_sep, contours=contours
            )

        if self.characteristics is not None:
            # Compute lines
            h_lines, v_lines = detect_lines(
                img=self.img,
                contours=self.characteristics.contours,
                char_length=self.characteristics.char_length,
                min_line_length=self.characteristics.min_line_length,
            )
            self.lines = h_lines + v_lines

    @property
    def horizontal_lines(self) -> list[Line]:
        return [line for line in self.lines if line.horizontal]

    @property
    def vertical_lines(self) -> list[Line]:
        return [line for line in self.lines if line.vertical]

    def extract_bordered_tables(
        self, implicit_rows: bool = False, implicit_columns: bool = False
    ) -> None:
        """
        Identify and extract bordered tables from image
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :return:
        """
        if self.characteristics is None:
            return

        self.tables = extract_bordered_tables(
            contours=self.characteristics.contours,
            horizontal_lines=self.horizontal_lines,
            vertical_lines=self.vertical_lines,
            char_length=self.characteristics.char_length,
            implicit_rows=implicit_rows,
            implicit_columns=implicit_columns,
        )

        # Post filter bordered tables
        self.tables = [tb for tb in self.tables if tb.has_valid_shape()]

    def extract_borderless_tables(self) -> None:
        """
        Identify and extract borderless tables from image
        :return:
        """
        if self.characteristics is None:
            return

        # Recompute thresholded image if relevant
        if not 0.5 < self.characteristics.char_length / 11 < 2:
            self.thresh = threshold_dark_areas(
                img=self.img, char_length=self.characteristics.char_length
            )

        # Extract borderless tables
        self.tables += extract_borderless_tables(
            thresh=self.thresh,
            char_length=self.characteristics.char_length,
            lines=self.lines,
            existing_tables=self.tables,
        )

    def extract_tables(
        self,
        implicit_rows: bool = False,
        implicit_columns: bool = False,
        borderless_tables: bool = False,
    ) -> list[Table]:
        """
        Identify and extract tables from image
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :return: list of identified tables
        """
        if self.characteristics is None:
            return self.tables

        # Extract bordered tables
        self.extract_bordered_tables(implicit_rows=implicit_rows, implicit_columns=implicit_columns)

        if borderless_tables:
            # Extract borderless tables
            self.extract_borderless_tables()

        # Compute title areas for tables
        return get_title_areas(
            contours=self.characteristics.contours,
            tables=self.tables,
            median_line_sep=self.characteristics.median_line_sep,
        )
