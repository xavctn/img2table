# coding: utf-8
from typing import Union, List

from img2table.objects.tables import Row, Cell, Line
from img2table.utils.common import is_contained_cell


def intersection_bbox_line(row: Union[Row, Cell], line: Line, horizontal_margin: int = 0) -> bool:
    """
    Determine if a line is intersecting a bounding box
    :param row: Row or Cell object
    :param line: Line object
    :param horizontal_margin: horizontal margin around row used to detect intersecting vertical lines
    :return: boolean indicating if there is an intersection between the bounding box and the line
    """
    # Check horizontal correspondence
    if not row.x1 - horizontal_margin <= line.x1 <= row.x2 + horizontal_margin:
        return False

    # Check vertical correspondence
    overlapping_pixels = max(0, min(row.y2, line.y2) - max(row.y1, line.y1))

    # Return intersection if the line intersects on at least 80% of the bounding box
    return overlapping_pixels / row.height > 0.8


def get_cells_h_line(line: Line, horizontal_lines: List[Line], vertical_lines: List[Line]) -> List[Cell]:
    """
    Identify all cells in image that uses an horizontal line as the upper bound of the cell
    :param line: horizontal line used in cells as the upper bound
    :param horizontal_lines: list of all horizontal lines
    :param vertical_lines: list of all vertical lines
    :return: list of cells that can be created from the line
    """
    # Create list of found cells
    output_cells = list()

    # Find horizontal lines below the current line which position matches with the line
    matching_lines = list()
    for h_line in [h_line for h_line in horizontal_lines if h_line.y1 > line.y1 + 10]:
        # Compute if right or left ends correspond in both lines
        l_corresponds = abs((line.x1 - h_line.x1) / line.width) <= 0.02
        r_corresponds = abs((line.x2 - h_line.x2) / line.width) <= 0.02
        # Compute if left and right ends of each table is coherent with the other line
        l_contained = (line.x1 <= h_line.x1 <= line.x2) or (h_line.x1 <= line.x1 <= h_line.x2)
        r_contained = (line.x1 <= h_line.x2 <= line.x2) or (h_line.x1 <= line.x2 <= h_line.x2)

        # If the h_line corresponds to the line, add it to the list of matching lines
        if (l_corresponds and r_corresponds) or (l_corresponds and r_contained) or (l_contained and r_corresponds) or \
                (l_contained and r_contained):
            matching_lines.append(h_line)

    # Loop over matching lines to check if there are vertical lines between it and the line in order to create a cell
    for h_line in matching_lines:
        # Create a cell from line and h_line
        h_line_cell = Cell.from_h_lines(line_1=line, line_2=h_line, minimal=True)
        # Get vertical lines that crosses the cell
        crossing_v_lines = [v_line for v_line in vertical_lines
                            if intersection_bbox_line(row=h_line_cell,
                                                      line=v_line,
                                                      horizontal_margin=max(round(0.05 * h_line_cell.width), 5))
                            ]

        # If there are some crossing lines, create cells based on those lines
        if len(crossing_v_lines) > 1:
            line_values = sorted([line.x1 for line in crossing_v_lines])
            boundaries = [bound for bound in zip(line_values, line_values[1:])]

            for boundary in boundaries:
                cell = Cell(x1=boundary[0], x2=boundary[1], y1=h_line_cell.y1, y2=h_line_cell.y2)
                output_cells.append(cell)

    output_cells = [c for c in output_cells if c.width >= 20]

    return output_cells


def adjacent_cells(cell_1: Cell, cell_2: Cell) -> bool:
    """
    Compute if two cells are adjacent
    :param cell_1: first cell object
    :param cell_2: second cell object
    :return: boolean indicating if cells are adjacent
    """
    # Check correspondence on vertical borders
    overlapping_y = len(list(set(list(range(cell_1.y1, cell_1.y2))) & set(list(range(cell_2.y1, cell_2.y2)))))
    diff_x = min(abs(cell_1.x2 - cell_2.x1),
                 abs(cell_1.x1 - cell_2.x2),
                 abs(cell_1.x1 - cell_2.x1),
                 abs(cell_1.x2 - cell_2.x2))
    if overlapping_y > 5 and diff_x / max(cell_1.width, cell_2.width) <= 0.05:
        return True

    # Check correspondence on horizontal borders
    overlapping_x = len(list(set(list(range(cell_1.x1, cell_1.x2))) & set(list(range(cell_2.x1, cell_2.x2)))))
    diff_y = min(abs(cell_1.y2 - cell_2.y1),
                 abs(cell_1.y1 - cell_2.y2),
                 abs(cell_1.y1 - cell_2.y1),
                 abs(cell_1.y2 - cell_2.y2))
    if overlapping_x > 5 and diff_y / max(cell_1.height, cell_2.height) <= 0.05:
        return True

    return False


def is_redundant(cell_1: Cell, cell_2: Cell) -> bool:
    """
    Determine if cell_2 is redundant with cell_1
    :param cell_1: first cell object
    :param cell_2: second cell object
    :return: boolean indicating if the second cell is redundant with the first one
    """
    # cell_2 is deemed as redundant if it covers at least 90% of cell_1 area and is adjacent to it
    contains = is_contained_cell(inner_cell=cell_1, outer_cell=cell_2, percentage=0.9)
    is_adjacent = adjacent_cells(cell_1=cell_1, cell_2=cell_2)
    return contains and is_adjacent


def deduplicate_cells(cells: List[Cell]) -> List[Cell]:
    """
    Deduplicate list of cells by removing redundant ones
    :param cells: list of cells
    :return: deduplicated list of cells
    """
    # Sort cells by area
    sorted_cells = sorted(cells, key=lambda x: x.height * x.width)

    final_cells = list()
    # For each cell, add it if it is not redundant with existing ones
    for idx, cell in enumerate(sorted_cells):
        if idx == 0:
            final_cells.append(cell)
        else:
            if not max([is_redundant(f_cell, cell) for f_cell in final_cells]):
                final_cells.append(cell)

    return final_cells


def get_cells(horizontal_lines: List[Line], vertical_lines: List[Line]) -> List[Cell]:
    """
    Identify cells from horizontal and vertical lines
    :param horizontal_lines: list of horizontal lines
    :param vertical_lines: list of vertical lines
    :return: list of all cells in image
    """
    # Retrieve all cells based on each horizontal line
    output_cells = list()
    for h_line in horizontal_lines:
        h_line_cells = get_cells_h_line(line=h_line,
                                        horizontal_lines=horizontal_lines,
                                        vertical_lines=vertical_lines)
        output_cells += h_line_cells

    # Deduplicate cells and identify relevant rectangles
    deduplicated_cells = deduplicate_cells(cells=output_cells)

    return deduplicated_cells
