import cv2

from img2table.tables.image import TableImage


def test_table_image() -> None:
    img = cv2.imread("test_data/test.png")
    assert img is not None
    img = 255 - cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    tb_image = TableImage(img=img, min_confidence=50)

    result = tb_image.extract_tables(implicit_rows=True)
    result = sorted(result, key=lambda tb: tb.x1 + tb.x2)

    assert (result[0].x1, result[0].y1, result[0].x2, result[0].y2) == (36, 22, 770, 328)
    assert (result[0].nb_rows, result[0].nb_columns) == (6, 3)

    assert (result[1].x1, result[1].y1, result[1].x2, result[1].y2) == (962, 22, 1155, 124)
    assert (result[1].nb_rows, result[1].nb_columns) == (2, 2)
