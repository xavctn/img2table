# coding: utf-8

from img2table.tables.objects.line import Line


def test_line():
    line = Line(x1=0, y1=20, x2=46, y2=73)

    assert round(line.angle) == 49
    assert line.width == 46
    assert line.height == 53
    assert round(line.length) == 70
    assert not line.vertical
    assert not line.horizontal


def test_reprocess_line():
    line = Line(x1=20, y1=73, x2=19, y2=20)

    reprocessed_line = line.reprocess()
    assert reprocessed_line == Line(x1=20, x2=20, y1=20, y2=73)
    assert reprocessed_line.vertical
