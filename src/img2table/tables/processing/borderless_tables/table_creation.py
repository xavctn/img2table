# coding: utf-8
from typing import List, Optional

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.common import is_contained_cell


def get_column_delimiters(tb_clusters: List[List[Cell]], margin: int = 5) -> List[int]:
    """
    Identify column delimiters from clusters
    :param tb_clusters: list of clusters composing the table
    :param margin: margin used for extremities
    :return: list of column delimiters
    """
    # Compute horizontal bounds of each cluster
    bounds = [(min([c.x1 for c in cluster]), max([c.x2 for c in cluster])) for cluster in tb_clusters]

    # Group clusters that corresponds to the same column
    col_clusters = list()
    for i in range(len(bounds)):
        for j in range(i, len(bounds)):
            # If clusters overlap, put them in same column
            x_diff = min(bounds[i][1], bounds[j][1]) - max(bounds[i][0], bounds[j][0])
            overlap = x_diff / max(bounds[i][1] - bounds[i][0], bounds[j][1] - bounds[j][0]) >= 0.1
            if overlap:
                matching_clusters = [idx for idx, cl in enumerate(col_clusters) if {i, j}.intersection(cl)]
                if matching_clusters:
                    remaining_clusters = [cl for idx, cl in enumerate(col_clusters) if idx not in matching_clusters]
                    new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(col_clusters) if idx in matching_clusters])
                    col_clusters = remaining_clusters + [new_cluster]
                else:
                    col_clusters.append({i, j})

    # Compute column bounds
    col_bounds = sorted([(min([bounds[i][0] for i in col]), max([bounds[i][1] for i in col])) for col in col_clusters],
                        key=lambda x: sum(x))

    # Create delimiters
    x_delimiters = [round((left[1] + right[0]) / 2) for left, right in zip(col_bounds, col_bounds[1:])]
    x_delimiters = [col_bounds[0][0] - margin] + x_delimiters + [col_bounds[-1][1] + margin]

    return x_delimiters


def get_row_delimiters(tb_clusters: List[List[Cell]], margin: int = 5) -> List[int]:
    """
    Identify row delimiters from clusters
    :param tb_clusters: list of clusters composing the table
    :param margin: margin used for extremities
    :return: list of row delimiters
    """
    # Get all cells from clusters
    cells = sorted([cell for cl in tb_clusters for cell in cl],
                   key=lambda c: c.y1)

    # Compute row clusters
    seq = iter(cells)
    row_clusters = [[next(seq)]]
    for cell in seq:
        cl_y1, cl_y2 = min([c.y1 for c in row_clusters[-1]]), max([c.y2 for c in row_clusters[-1]])
        y_corr = min(cell.y2, cl_y2) - max(cell.y1, cl_y1)
        if y_corr / max(cl_y2 - cl_y1, cell.y2 - cell.y1) <= 0.2:
            row_clusters.append([])
        row_clusters[-1].append(cell)

    # Compute row bounds
    row_bounds = [(min([c.y1 for c in row]), max([c.y2 for c in row])) for row in row_clusters]

    # Create delimiters
    y_delimiters = [round((up[1] + down[0]) / 2) for up, down in zip(row_bounds, row_bounds[1:])]
    y_delimiters = [row_bounds[0][0] - margin] + y_delimiters + [row_bounds[-1][1] + margin]

    return y_delimiters


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


def create_table_from_clusters(tb_clusters: List[List[Cell]], segment_cells: List[Cell]) -> Optional[Table]:
    """
    Create table from aligned clusters forming it
    :param tb_clusters: list of aligned word clusters
    :param segment_cells: list of word cells comprised in image segment
    :return: Table object if relevant
    """
    # Identify column delimiters from clusters
    col_dels = get_column_delimiters(tb_clusters=tb_clusters)

    # Identify row delimiters from clusters
    row_dels = get_row_delimiters(tb_clusters=tb_clusters)

    # Create rows
    list_rows = list()
    for upper_bound, lower_bound in zip(row_dels, row_dels[1:]):
        l_cells = list()
        for l_bound, r_bound in zip(col_dels, col_dels[1:]):
            l_cells.append(Cell(x1=l_bound, y1=upper_bound, x2=r_bound, y2=lower_bound))
        list_rows.append(Row(cells=l_cells))

    # Create table
    table = Table(rows=list_rows)

    # Check coherency of top and bottom rows of table which should have at least 2 columns with content
    coherent_table = check_coherency_rows(tb=table,
                                          word_cells=[cell for cl in tb_clusters for cell in cl])

    # Check if table is mainly comprised of cells used for its creation
    final_table = check_versus_content(table=coherent_table,
                                       table_cells=[cell for cl in tb_clusters for cell in cl],
                                       segment_cells=segment_cells)

    return final_table
