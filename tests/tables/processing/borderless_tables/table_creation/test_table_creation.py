# coding: utf-8
import json

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.row import Row
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.table_creation import create_table_from_clusters


def test_create_table_from_clusters():
    with open("test_data/clusters.json", "r") as f:
        clusters = [[Cell(**element) for element in cl] for cl in json.load(f)]

    with open("test_data/segment_cells.json", "r") as f:
        segment_cells = [Cell(**element) for element in json.load(f)]

    result = create_table_from_clusters(tb_clusters=clusters, segment_cells=segment_cells)

    expected = Table(rows=[Row(cells=[Cell(x1=35, y1=83, x2=218, y2=124),
                                      Cell(x1=218, y1=83, x2=447, y2=124),
                                      Cell(x1=447, y1=83, x2=621, y2=124)]),
                           Row(cells=[Cell(x1=35, y1=124, x2=218, y2=176),
                                      Cell(x1=218, y1=124, x2=447, y2=176),
                                      Cell(x1=447, y1=124, x2=621, y2=176)]),
                           Row(cells=[Cell(x1=35, y1=176, x2=218, y2=226),
                                      Cell(x1=218, y1=176, x2=447, y2=226),
                                      Cell(x1=447, y1=176, x2=621, y2=226)]),
                           Row(cells=[Cell(x1=35, y1=226, x2=218, y2=277),
                                      Cell(x1=218, y1=226, x2=447, y2=277),
                                      Cell(x1=447, y1=226, x2=621, y2=277)]),
                           Row(cells=[Cell(x1=35, y1=277, x2=218, y2=319),
                                      Cell(x1=218, y1=277, x2=447, y2=319),
                                      Cell(x1=447, y1=277, x2=621, y2=319)])
                           ])

    assert result == expected
