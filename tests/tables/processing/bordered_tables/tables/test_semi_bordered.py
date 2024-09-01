# coding: utf-8

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.processing.bordered_tables.tables.semi_bordered import get_lines_in_cluster, \
    add_semi_bordered_cells, identify_table_dimensions, identify_potential_new_cells, update_cluster_cells


def test_get_lines_in_cluster():
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    lines = [Line(x1=50, x2=205, y1=100, y2=100),
             Line(x1=50, x2=205, y1=200, y2=200),
             Line(x1=100, x2=100, y1=30, y2=270),
             Line(x1=200, x2=200, y1=30, y2=270)]

    h_lines_cl, v_lines_cl = get_lines_in_cluster(cluster=cluster, lines=lines)

    assert h_lines_cl == [Line(x1=50, x2=205, y1=100, y2=100),
                          Line(x1=50, x2=205, y1=200, y2=200)]
    assert v_lines_cl == [Line(x1=100, x2=100, y1=30, y2=270),
                          Line(x1=200, x2=200, y1=30, y2=270)]


def test_identify_table_dimensions():
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    h_lines_cl = [Line(x1=50, x2=205, y1=100, y2=100),
                  Line(x1=50, x2=205, y1=200, y2=200)]
    v_lines_cl = [Line(x1=100, x2=100, y1=30, y2=270),
                  Line(x1=200, x2=200, y1=30, y2=270)]

    left_val, right_val, top_val, bottom_val = identify_table_dimensions(cluster=cluster,
                                                                         h_lines_cl=h_lines_cl,
                                                                         v_lines_cl=v_lines_cl,
                                                                         char_length=5)

    assert (left_val, right_val, top_val, bottom_val) == (50, 200, 30, 270)


def test_identify_potential_new_cells():
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    h_lines_cl = [Line(x1=50, x2=205, y1=100, y2=100),
                  Line(x1=50, x2=205, y1=200, y2=200)]
    v_lines_cl = [Line(x1=100, x2=100, y1=30, y2=270),
                  Line(x1=200, x2=200, y1=30, y2=270)]
    left_val, right_val, top_val, bottom_val = 50, 200, 30, 270

    result = identify_potential_new_cells(cluster=cluster,
                                          h_lines_cl=h_lines_cl,
                                          v_lines_cl=v_lines_cl,
                                          left_val=left_val,
                                          right_val=right_val,
                                          top_val=top_val,
                                          bottom_val=bottom_val)

    expected = [Cell(x1=100, y1=200, x2=200, y2=270),
                Cell(x1=50, y1=30, x2=100, y2=100),
                Cell(x1=50, y1=100, x2=100, y2=200),
                Cell(x1=100, y1=100, x2=200, y2=200),
                Cell(x1=100, y1=30, x2=200, y2=100),
                Cell(x1=50, y1=200, x2=100, y2=270)]

    assert sorted(result, key=lambda c: c.bbox()) == sorted(expected, key=lambda c: c.bbox())


def test_update_cluster_cells():
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    new_cells = [Cell(x1=100, y1=200, x2=200, y2=270),
                 Cell(x1=50, y1=30, x2=100, y2=100),
                 Cell(x1=50, y1=100, x2=100, y2=200),
                 Cell(x1=100, y1=100, x2=200, y2=200),
                 Cell(x1=100, y1=30, x2=200, y2=100),
                 Cell(x1=50, y1=200, x2=100, y2=270)]

    result = update_cluster_cells(cluster=cluster, new_cells=new_cells)

    expected = [Cell(x1=100, y1=100, x2=200, y2=200),
                Cell(x1=50, y1=200, x2=100, y2=270),
                Cell(x1=100, y1=30, x2=200, y2=100),
                Cell(x1=50, y1=30, x2=100, y2=100),
                Cell(x1=100, y1=200, x2=200, y2=270),
                Cell(x1=50, y1=100, x2=100, y2=200)]

    assert sorted(result, key=lambda c: c.bbox()) == sorted(expected, key=lambda c: c.bbox())


def test_add_semi_bordered_cells():
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    lines = [Line(x1=50, x2=205, y1=100, y2=100),
             Line(x1=50, x2=205, y1=200, y2=200),
             Line(x1=100, x2=100, y1=30, y2=270),
             Line(x1=200, x2=200, y1=30, y2=270)]

    result = add_semi_bordered_cells(cluster=cluster,
                                     lines=lines,
                                     char_length=5)

    expected = [Cell(x1=100, y1=100, x2=200, y2=200),
                Cell(x1=50, y1=200, x2=100, y2=270),
                Cell(x1=100, y1=30, x2=200, y2=100),
                Cell(x1=50, y1=30, x2=100, y2=100),
                Cell(x1=100, y1=200, x2=200, y2=270),
                Cell(x1=50, y1=100, x2=100, y2=200)]

    assert sorted(result, key=lambda c: c.bbox()) == sorted(expected, key=lambda c: c.bbox())
