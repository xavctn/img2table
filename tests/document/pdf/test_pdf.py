from io import BytesIO
from pathlib import Path

import pytest

from img2table._validation import ValidationError
from img2table.document.pdf import PDF
from img2table.tables.extraction import BBox


def test_validators() -> None:
    with pytest.raises(ValidationError):
        PDF(src=1)  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValidationError):
        PDF(src="img", pages=12)  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValidationError):
        PDF(src="img", pages=["True"])  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValidationError):
        PDF(src="img", pages=[1], detect_rotation="a")  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValidationError):
        PDF(src="img", pages=(0, 1))  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValidationError):
        PDF(src="img", pages=[True])

    with pytest.raises(ValidationError):
        PDF(src="img", pdf_text_extraction="a")  # ty:ignore[invalid-argument-type]

    with pytest.raises(ValidationError):
        PDF(src="img").extract_tables(max_workers=0)

    with pytest.raises(ValidationError):
        PDF(src="img").extract_tables(max_workers=True)  # ty:ignore[invalid-argument-type]


def test_load_pdf() -> None:
    # Load from path
    pdf_from_path = PDF(src="test_data/test.pdf")

    # Load from bytes
    with Path("test_data/test.pdf").open("rb") as f:
        pdf_from_bytes = PDF(src=f.read())

    # Load from BytesIO
    with Path("test_data/test.pdf").open("rb") as f:
        pdf_from_bytesio = PDF(src=BytesIO(f.read()))

    assert pdf_from_path.file_bytes == pdf_from_bytes.file_bytes == pdf_from_bytesio.file_bytes

    assert next(iter(pdf_from_path.images)).shape == (2200, 1700, 3)


def test_pdf_pages() -> None:
    assert len(list(PDF(src="test_data/test.pdf").images)) == 2
    assert len(list(PDF(src="test_data/test.pdf", pages=[0]).images)) == 1


def test_pdf_tables() -> None:
    pdf = PDF(src="test_data/test.pdf")

    result = pdf.extract_tables(implicit_rows=True, min_confidence=50)

    assert result[0][0].title == "Example of Data Table 1"
    assert result[0][0].bbox == BBox(x1=236, y1=250, x2=1443, y2=544)
    assert (len(result[0][0].content), len(result[0][0].content[0])) == (5, 4)

    assert result[0][1].title == "Example of Data Table 2"
    assert result[0][1].bbox == BBox(x1=236, y1=672, x2=1452, y2=972)
    assert (len(result[0][1].content), len(result[0][1].content[0])) == (5, 4)

    assert result[1][0].title == "Example of Data Table 3"
    assert result[1][0].bbox == BBox(x1=236, y1=250, x2=1443, y2=544)
    assert (len(result[1][0].content), len(result[1][0].content[0])) == (5, 4)

    assert result[1][1].title == "Example of Data Table 4"
    assert result[1][1].bbox == BBox(x1=236, y1=672, x2=1452, y2=972)
    assert (len(result[1][1].content), len(result[1][1].content[0])) == (5, 4)


def test_pdf_tables_parallel() -> None:
    serial_result = PDF(src="test_data/test.pdf").extract_tables(
        implicit_rows=True,
        min_confidence=50,
        max_workers=1,
    )
    parallel_result = PDF(src="test_data/test.pdf").extract_tables(
        implicit_rows=True,
        min_confidence=50,
        max_workers=2,
    )

    assert serial_result.keys() == parallel_result.keys()

    for page in serial_result:
        assert len(serial_result[page]) == len(parallel_result[page])
        for serial_table, parallel_table in zip(
            serial_result[page], parallel_result[page], strict=True
        ):
            assert serial_table.title == parallel_table.title
            assert serial_table.bbox == parallel_table.bbox
            assert serial_table.html == parallel_table.html
