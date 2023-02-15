# coding: utf-8
from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.alignment import left_aligned, right_aligned, center_aligned, \
    vertically_coherent_cluster, cluster_aligned_text


def test_left_aligned():
    cell_1 = Cell(x1=10, y1=0, x2=25, y2=10)
    cell_2 = Cell(x1=11, y1=10, x2=33, y2=20)
    cell_3 = Cell(x1=25, y1=20, x2=35, y2=30)

    assert left_aligned(cell_1=cell_1, cell_2=cell_2)
    assert not left_aligned(cell_1=cell_1, cell_2=cell_3)
    assert not left_aligned(cell_1=cell_2, cell_2=cell_3)


def test_right_aligned():
    cell_1 = Cell(x1=10, y1=0, x2=25, y2=10)
    cell_2 = Cell(x1=11, y1=10, x2=33, y2=20)
    cell_3 = Cell(x1=0, y1=20, x2=35, y2=30)

    assert not right_aligned(cell_1=cell_1, cell_2=cell_2)
    assert not right_aligned(cell_1=cell_1, cell_2=cell_3)
    assert right_aligned(cell_1=cell_2, cell_2=cell_3)


def test_center_aligned():
    cell_1 = Cell(x1=10, y1=0, x2=25, y2=10)
    cell_2 = Cell(x1=11, y1=10, x2=33, y2=20)
    cell_3 = Cell(x1=0, y1=20, x2=35, y2=30)

    assert not center_aligned(cell_1=cell_1, cell_2=cell_2)
    assert center_aligned(cell_1=cell_1, cell_2=cell_3)
    assert not center_aligned(cell_1=cell_2, cell_2=cell_3)


def test_vertically_coherent_cluster():
    cluster = [Cell(x1=0, x2=20, y1=0, y2=10),
               Cell(x1=0, x2=20, y1=20, y2=30),
               Cell(x1=0, x2=20, y1=40, y2=50),
               Cell(x1=0, x2=20, y1=90, y2=100),
               Cell(x1=0, x2=20, y1=200, y2=210),
               Cell(x1=0, x2=20, y1=220, y2=230),
               ]
    v_clusters = vertically_coherent_cluster(cluster=cluster)

    assert v_clusters == [
        [Cell(x1=0, x2=20, y1=0, y2=10),
         Cell(x1=0, x2=20, y1=20, y2=30),
         Cell(x1=0, x2=20, y1=40, y2=50)],
        [Cell(x1=0, x2=20, y1=200, y2=210),
         Cell(x1=0, x2=20, y1=220, y2=230)]
    ]


def test_cluster_aligned_text():
    segment = [Cell(x1=0, x2=20, y1=0, y2=10),
               Cell(x1=0, x2=20, y1=20, y2=30),
               Cell(x1=0, x2=20, y1=40, y2=50),
               Cell(x1=0, x2=20, y1=90, y2=100),
               Cell(x1=0, x2=20, y1=200, y2=210),
               Cell(x1=0, x2=20, y1=220, y2=230),
               Cell(x1=100, x2=110, y1=220, y2=230),
               Cell(x1=223, x2=300, y1=0, y2=100),
               Cell(x1=108, x2=302, y1=220, y2=230),
               Cell(x1=112, x2=296, y1=340, y2=350),
               ]

    aligned_text = cluster_aligned_text(segment=segment)

    assert aligned_text == [
        [Cell(x1=0, x2=20, y1=0, y2=10),
         Cell(x1=0, x2=20, y1=20, y2=30),
         Cell(x1=0, x2=20, y1=40, y2=50)],
        [Cell(x1=0, x2=20, y1=200, y2=210),
         Cell(x1=0, x2=20, y1=220, y2=230)],
        [Cell(x1=223, x2=300, y1=0, y2=100),
         Cell(x1=108, x2=302, y1=220, y2=230),
         Cell(x1=112, x2=296, y1=340, y2=350)]
    ]
