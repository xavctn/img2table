
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table


def merge_consecutive_tables(tables: list[Table], contours: list[Cell]) -> list[Table]:
    """
    Merge consecutive coherent tables
    :param tables: list of detected tables
    :param contours: list of image contours
    :return: list of processed tables
    """
    if len(tables) == 0:
        return []

    # Create table clusters
    seq = iter(sorted(tables, key=lambda t: t.y1))
    clusters = [[next(seq)]]

    for tb in seq:
        prev_table = clusters[-1][-1]
        # Check if there are elements between the two tables
        in_between_contours = [c for c in contours if c.y1 >= prev_table.y2 and c.y2 <= tb.y1
                               and c.x2 >= min(prev_table.x1, tb.x1)
                               and c.x1 <= max(prev_table.x2, tb.x2)]
        # Check coherency of tables
        prev_tb_cols = sorted([ln for ln in prev_table.lines if ln.vertical], key=lambda ln: ln.x1)
        tb_cols = sorted([ln for ln in tb.lines if ln.vertical], key=lambda ln: ln.x1)
        coherency_lines = all(abs(l1.x1 - l2.x1) <= 2 for l1, l2 in zip(prev_tb_cols, tb_cols))

        if not (len(in_between_contours) == 0 and prev_table.nb_columns == tb.nb_columns and coherency_lines):
            clusters.append([])
        clusters[-1].append(tb)

    # Create merged tables
    merged_tables = []
    for cl in clusters:
        if len(cl) == 1:
            merged_tables += cl
        else:
            # Create new table
            new_tb = Table(rows=[row for tb in cl for row in tb.items], borderless=False)
            merged_tables.append(new_tb)

    return merged_tables
