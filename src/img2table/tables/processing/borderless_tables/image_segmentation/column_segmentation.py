# coding: utf-8
import operator
import random
from dataclasses import dataclass
from functools import partial
from queue import PriorityQueue
from typing import List

import cv2
import numpy as np

from img2table.tables import cluster_items
from img2table.tables.objects.cell import Cell


@dataclass
class ColumnGroup:
    columns: List[Cell]

    @property
    def x1(self):
        return min([c.x1 for c in self.columns])

    @property
    def y1(self):
        return min([c.y1 for c in self.columns])

    @property
    def x2(self):
        return max([c.x2 for c in self.columns])

    @property
    def y2(self):
        return max([c.y2 for c in self.columns])

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def segments(self) -> List[Cell]:
        # Sort columns
        sorted_cols = sorted(self.columns, key=lambda c: c.x1 + c.x2)

        # Get x values
        x_values = [int((c_left.x2 + c_right.x1) / 2) for c_left, c_right in zip(sorted_cols, sorted_cols[1:])]
        x_values = [self.x1] + x_values + [self.x2]

        return [Cell(x1=x_left, y1=self.y1, x2=x_right, y2=self.y2) for x_left, x_right in zip(x_values, x_values[1:])]

    def __eq__(self, other):
        if isinstance(other, ColumnGroup):
            try:
                assert self.columns == other.columns
                return True
            except AssertionError:
                return False
        return False


@dataclass
class Rectangle:
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_cell(cls, cell: Cell) -> "Rectangle":
        return cls(x1=cell.x1, y1=cell.y1, x2=cell.x2, y2=cell.y2)

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def distance(self, other):
        return (self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2

    def overlaps(self, other):
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        return max(x_right - x_left, 0) * max(y_bottom - y_top, 0) > 0


def same_column(median_line_sep: float, cnt_1: Cell, cnt_2: Cell) -> bool:
    """
    Identify if two contours can be part of the same column
    :param median_line_sep: median line separation
    :param cnt_1: first contour
    :param cnt_2: second contour
    :return: boolean indicating if two contours can be part of the same column
    """
    # Compute x overlap
    x_overlap = min(cnt_1.x2, cnt_2.x2) - max(cnt_1.x1, cnt_2.x1)
    x_matches = x_overlap / max(cnt_1.width, cnt_2.width) >= 0.8

    # Compute y diff
    y_diff = min(abs(cnt_1.y2 - cnt_2.y1), abs(cnt_2.y2 - cnt_1.y1))
    y_matches = y_diff <= 3 * median_line_sep

    return x_matches and y_matches


def coherent_columns(col_1: Cell, col_2: Cell) -> bool:
    """
    Identify if two columns are coherent
    :param col_1: first column contour
    :param col_2: second column contour
    :return: boolean indicating if two columns are coherent
    """
    width_matches = min(col_1.width, col_2.width) / max(col_1.width, col_2.width) >= 0.8
    top_matches = abs(col_1.y1 - col_2.y1) / max(col_1.height, col_2.height) <= 0.05

    return width_matches and top_matches


def intertwined_col_groups(col_gp: ColumnGroup, columns: List[Cell]) -> bool:
    """
    Identify if a column is intertwined within a column group
    :param col_gp: column group
    :param columns: list of columns
    :return: boolean if a column is intertwined within a column group
    """
    # Get columns that do not belong to column group
    other_cols = [col for col in columns if col not in col_gp.columns]

    for col in other_cols:
        # Check if column is intertwined
        y_overlap = min(col_gp.y2, col.y2) - max(col_gp.y1, col.y1)
        y_matches = y_overlap / min(col_gp.height, col.height) >= 0.1

        x_overlap = min(col_gp.x2, col.x2) - max(col_gp.x1, col.x1)
        x_matches = x_overlap / min(col_gp.width, col.width) >= 0.2

        if x_matches and y_matches:
            return True

    return False


def get_image_columns(img: np.ndarray, char_length: float, median_line_sep: float) -> List[ColumnGroup]:
    """
    Identify group of columns in image
    :param img: image array
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: list of column groups
    """
    # Reprocess images
    blur = cv2.medianBlur(img, 3)
    thresh = cv2.Canny(blur, 0, 0)

    # Define kernel by using median line separation and character length
    kernel_size = (round(char_length * 3), round(1.25 * median_line_sep))

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Create new image by adding black borders
    margin = 10
    black_borders = np.zeros(tuple(map(operator.add, img.shape, (2 * margin, 2 * margin))), dtype=np.uint8)
    black_borders[margin:img.shape[0] + margin, margin:img.shape[1] + margin] = dilate

    # Find contours
    cnts = cv2.findContours(black_borders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Get list of contours
    list_cnts_cell = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x = x - margin
        y = y - margin
        contour_cell = Cell(x, y, x + w, y + h)
        list_cnts_cell.append(contour_cell)

    # Identify elements that can be part of the same column
    clustering_f = partial(same_column, median_line_sep)
    columns = [Cell(x1=min([c.x1 for c in col_gp]),
                    y1=min([c.y1 for c in col_gp]),
                    x2=max([c.x2 for c in col_gp]),
                    y2=max([c.y2 for c in col_gp]))
               for col_gp in cluster_items(items=list_cnts_cell, clustering_func=clustering_f)]

    # Filter columns that represent at least 1/3 of the height of the image
    columns = [col for col in columns if col.height >= 0.33 * img.shape[0]]

    # Identify columns that match together
    column_groups = cluster_items(items=columns, clustering_func=coherent_columns)
    column_groups = [ColumnGroup(columns=gp) for gp in column_groups if 3 >= len(gp) >= 2]

    # Check coherency of column groups (no intertwined groups)
    coherent_col_groups = [col_gp for col_gp in column_groups
                           if not intertwined_col_groups(col_gp=col_gp, columns=columns)
                           and col_gp.height >= 0.5 * img.shape[0]
                           and col_gp.width >= 0.66 * img.shape[1]]

    return coherent_col_groups


def identify_remaining_segments(existing_segments: List[Cell], height: int, width: int) -> List[Cell]:
    """
    Identify remaining segments in image
    :param existing_segments: list of existing image segments
    :param height: image height
    :param width: image width
    :return: list of whitespaces as Cell objects
    """
    # Create rectangle objects from inputs
    group_rect = Rectangle(x1=0, y1=0, x2=width, y2=height)
    obstacles = [Rectangle.from_cell(cell=element) for element in existing_segments]

    # Initiate queue
    queue = PriorityQueue()
    queue.put([-group_rect.area, group_rect, obstacles])

    segments = list()
    while not queue.qsize() == 0:
        q, r, obs = queue.get()
        if len(obs) == 0:
            # Update segments
            segments.append(r)

            # Update elements in queue
            for element in queue.queue:
                if element[1].overlaps(r):
                    element[2] += [r]

            continue

        # Get most pertinent obstacle
        pivot = sorted(obs, key=lambda o: o.distance(r))[0]

        # Create new rectangles
        rects = [Rectangle(x1=pivot.x2, y1=r.y1, x2=r.x2, y2=r.y2),
                 Rectangle(x1=r.x1, y1=r.y1, x2=pivot.x1, y2=r.y2),
                 Rectangle(x1=r.x1, y1=pivot.y2, x2=r.x2, y2=r.y2),
                 Rectangle(x1=r.x1, y1=r.y1, x2=r.x2, y2=pivot.y1)]

        for rect in rects:
            if rect.area > group_rect.area / 100:
                rect_obstacles = [o for o in obs if o.overlaps(rect)]
                queue.put([-rect.area + random.uniform(0, 1), rect, rect_obstacles])

    return [seg.cell for seg in segments]


def segment_image_columns(img: np.ndarray, char_length: float, median_line_sep: float,
                          contours: List[Cell]) -> List[Cell]:
    """
    Create image segments by identifying columns
    :param img: image array
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param contours: list of image contours
    :return: list of segments corresponding to image
    """
    # Get column groups
    column_groups = get_image_columns(img=img,
                                      char_length=char_length,
                                      median_line_sep=median_line_sep)

    # If no column group has been found, return a segment corresponding to the image
    if len(column_groups) == 0:
        return [Cell(x1=0, y1=0, x2=img.shape[1], y2=img.shape[0])]

    # Identify segments outside of columns
    top_segment = Cell(x1=0,
                       y1=0,
                       x2=img.shape[1],
                       y2=min([gp.y1 for gp in column_groups]))
    bottom_segment = Cell(x1=0,
                          y1=max([gp.y2 for gp in column_groups]),
                          x2=img.shape[1],
                          y2=img.shape[0])
    left_segment = Cell(x1=0,
                        y1=min([gp.y1 for gp in column_groups]),
                        x2=min([gp.x1 for gp in column_groups]),
                        y2=max([gp.y2 for gp in column_groups]))
    right_segment = Cell(x1=min([gp.x2 for gp in column_groups]),
                         y1=min([gp.y1 for gp in column_groups]),
                         x2=img.shape[1],
                         y2=max([gp.y2 for gp in column_groups]))

    # Create image segments
    img_segments = [top_segment, bottom_segment, left_segment, right_segment]
    img_segments += [seg for gp in column_groups for seg in gp.segments]

    # Identify remaining segments
    img_segments += identify_remaining_segments(existing_segments=img_segments,
                                                height=img.shape[0],
                                                width=img.shape[1])

    final_segments = list()
    for seg in img_segments:
        # Get included contours
        included_contours = [c for c in contours
                             if c.x1 >= seg.x1 and c.x2 <= seg.x2 and c.y1 >= seg.y1 and c.y2 <= seg.y2]

        if included_contours:
            final_segments.append(seg)

    return final_segments
