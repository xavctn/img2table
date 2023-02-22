# coding: utf-8

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.table_creation.coherency import check_versus_content


def test_check_versus_content():
    table = Table(rows=Row(cells=[Cell(x1=0, y1=0, x2=100, y2=100)]))
    table_cells = [Cell(x1=0, y1=0, x2=100, y2=100), Cell(x1=0, y1=0, x2=100, y2=100)]
    segment_cells = [Cell(x1=0, y1=0, x2=100, y2=100), Cell(x1=0, y1=0, x2=100, y2=100)]

    assert check_versus_content(table=table, table_cells=table_cells, segment_cells=segment_cells) == table

    # Add some segment cells
    segment_cells += segment_cells
    assert check_versus_content(table=table, table_cells=table_cells, segment_cells=segment_cells) is None
