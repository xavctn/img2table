from img2table.tables.bordered.tables.misc.semi_bordered import (
    add_semi_bordered_cells,
    get_cluster_characteristics,
    get_lines_in_cluster,
    identify_potential_new_cells,
    identify_table_dimensions,
    update_cluster_cells,
)
from img2table.tables.types import Cell, Line


def test_get_lines_in_cluster() -> None:
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    characteristics = get_cluster_characteristics(cluster=cluster, char_length=5)
    lines = [
        Line(x1=50, x2=205, y1=100, y2=100),
        Line(x1=50, x2=205, y1=200, y2=200),
        Line(x1=100, x2=100, y1=30, y2=270),
        Line(x1=200, x2=200, y1=30, y2=270),
        Line(x1=210, x2=400, y1=100, y2=100),
        Line(x1=100, x2=100, y1=300, y2=450),
    ]

    h_lines_cl, v_lines_cl = get_lines_in_cluster(lines=lines, characteristics=characteristics)

    assert h_lines_cl == [Line(x1=50, x2=205, y1=100, y2=100), Line(x1=50, x2=205, y1=200, y2=200)]
    assert v_lines_cl == [Line(x1=100, x2=100, y1=30, y2=270), Line(x1=200, x2=200, y1=30, y2=270)]


def test_identify_table_dimensions() -> None:
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    characteristics = get_cluster_characteristics(cluster=cluster, char_length=5)
    h_lines_cl = [
        Line(x1=50, x2=205, y1=100, y2=100),
        Line(x1=150, x2=205, y1=200, y2=200),
    ]
    v_lines_cl = [Line(x1=100, x2=100, y1=30, y2=270), Line(x1=200, x2=200, y1=30, y2=270)]

    left, right, top, bottom = identify_table_dimensions(
        characteristics=characteristics,
        h_lines_cl=h_lines_cl,
        v_lines_cl=v_lines_cl,
    )

    assert (left, right, top, bottom) == (100, 205, 30, 270)


def test_identify_table_dimensions_cluster_fallback() -> None:
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    characteristics = get_cluster_characteristics(cluster=cluster, char_length=5)
    h_lines_cl = [
        Line(x1=70, x2=180, y1=99, y2=99),
        Line(x1=50, x2=205, y1=101, y2=101),
        Line(x1=50, x2=205, y1=200, y2=200),
    ]
    v_lines_cl = [Line(x1=100, x2=100, y1=30, y2=270), Line(x1=200, x2=200, y1=30, y2=270)]

    left, right, top, bottom = identify_table_dimensions(
        characteristics=characteristics,
        h_lines_cl=h_lines_cl,
        v_lines_cl=v_lines_cl,
    )

    assert (left, right, top, bottom) == (100, 200, 30, 270)


def test_identify_potential_new_cells() -> None:
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    h_lines_cl = [Line(x1=50, x2=205, y1=100, y2=100), Line(x1=50, x2=205, y1=200, y2=200)]
    v_lines_cl = [Line(x1=100, x2=100, y1=30, y2=270), Line(x1=200, x2=200, y1=30, y2=270)]

    result = identify_potential_new_cells(
        cluster=cluster,
        h_lines_cl=h_lines_cl,
        v_lines_cl=v_lines_cl,
        left=50,
        right=205,
        top=30,
        bottom=270,
    )

    expected = [
        Cell(x1=50, y1=30, x2=100, y2=100),
        Cell(x1=50, y1=100, x2=100, y2=200),
        Cell(x1=50, y1=200, x2=100, y2=270),
        Cell(x1=100, y1=30, x2=200, y2=100),
        Cell(x1=100, y1=200, x2=200, y2=270),
        Cell(x1=200, y1=30, x2=205, y2=100),
        Cell(x1=200, y1=100, x2=205, y2=200),
        Cell(x1=200, y1=200, x2=205, y2=270),
    ]

    assert sorted(result, key=lambda c: c.bbox()) == sorted(expected, key=lambda c: c.bbox())


def test_update_cluster_cells() -> None:
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    new_cells = [
        Cell(x1=100, y1=200, x2=200, y2=270),
        Cell(x1=50, y1=30, x2=100, y2=100),
        Cell(x1=50, y1=100, x2=100, y2=200),
        Cell(x1=100, y1=30, x2=200, y2=100),
        Cell(x1=50, y1=200, x2=100, y2=270),
    ]

    result = update_cluster_cells(cluster=cluster, new_cells=new_cells, char_length=5)

    expected = [
        Cell(x1=100, y1=100, x2=200, y2=200),
        Cell(x1=50, y1=200, x2=100, y2=270),
        Cell(x1=100, y1=30, x2=200, y2=100),
        Cell(x1=50, y1=30, x2=100, y2=100),
        Cell(x1=100, y1=200, x2=200, y2=270),
        Cell(x1=50, y1=100, x2=100, y2=200),
    ]

    assert sorted(result, key=lambda c: c.bbox()) == sorted(expected, key=lambda c: c.bbox())


def test_update_cluster_cells_ignores_existing_cell() -> None:
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    new_cells = [Cell(x1=100, x2=200, y1=100, y2=200)]

    result = update_cluster_cells(cluster=cluster, new_cells=new_cells, char_length=5)

    assert result == cluster


def test_add_semi_bordered_cells() -> None:
    cluster = [Cell(x1=100, x2=200, y1=100, y2=200)]
    lines = [
        Line(x1=50, x2=210, y1=100, y2=100),
        Line(x1=50, x2=210, y1=200, y2=200),
        Line(x1=100, x2=100, y1=30, y2=270),
        Line(x1=200, x2=200, y1=30, y2=270),
    ]

    result = add_semi_bordered_cells(cluster=cluster, lines=lines, char_length=5)

    expected = [
        Cell(x1=100, y1=100, x2=200, y2=200),
        Cell(x1=50, y1=200, x2=100, y2=270),
        Cell(x1=100, y1=30, x2=200, y2=100),
        Cell(x1=50, y1=30, x2=100, y2=100),
        Cell(x1=100, y1=200, x2=200, y2=270),
        Cell(x1=50, y1=100, x2=100, y2=200),
        Cell(x1=200, y1=30, x2=210, y2=100),
        Cell(x1=200, y1=100, x2=210, y2=200),
        Cell(x1=200, y1=200, x2=210, y2=270),
    ]

    assert sorted(result, key=lambda c: c.bbox()) == sorted(expected, key=lambda c: c.bbox())
