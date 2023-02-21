# coding: utf-8
from typing import List, Optional

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
from img2table.tables.processing.common import is_contained_cell


def check_coherency_rows(tb: Table, word_cells: List[Cell]) -> Table:
    """
    Check coherency of top and bottom rows of table which should have at least 2 columns with content
    :param tb: Table object
    :param word_cells: list of word cells used to create the table
    :return:
    """
    # Check top row
    while tb.nb_rows > 0:
        # Get top row cells
        top_row_cells = [cell for cell in tb.items[0].items]

        # Check number of cells with content
        nb_cells_content = 0
        for cell in top_row_cells:
            if any([is_contained_cell(inner_cell=w_cell, outer_cell=cell, percentage=0.9) for w_cell in word_cells]):
                nb_cells_content += 1

        if nb_cells_content < 2:
            tb = Table(rows=tb.items[1:])
        else:
            break

    # Check bottom row
    while tb.nb_rows > 0:
        # Get bottom row cells
        bottom_row_cells = [cell for cell in tb.items[-1].items]

        # Check number of cells with content
        nb_cells_content = 0
        for cell in bottom_row_cells:
            if any([is_contained_cell(inner_cell=w_cell, outer_cell=cell, percentage=0.9) for w_cell in word_cells]):
                nb_cells_content += 1

        if nb_cells_content < 2:
            tb = Table(rows=tb.items[:-1])
        else:
            break

    return tb


def check_versus_content(table: Table, table_cells: List[Cell], segment_cells: List[Cell]) -> Optional[Table]:
    """
    Check if table is mainly comprised of cells used for its creation
    :param table: Table object
    :param table_cells: list of cells used for its creation
    :param segment_cells: list of all cells in segment
    :return: Table object if it is mainly comprised of cells used for its creation
    """
    # Check table attributes
    if table.nb_rows * table.nb_columns == 0:
        return None

    # Get table bbox as Cell
    tb_bbox = Cell(*table.bbox())

    # Get number of table cells included in table
    nb_tb_cells_included = len([cell for cell in table_cells
                                if is_contained_cell(inner_cell=cell, outer_cell=tb_bbox)])

    # Get number of segments cells included in table
    nb_seg_cells_included = len([cell for cell in segment_cells
                                 if is_contained_cell(inner_cell=cell, outer_cell=tb_bbox)])

    # Return table if table cells represent at least 80% of cells in bbox
    return table if nb_tb_cells_included > 0.8 * nb_seg_cells_included else None

