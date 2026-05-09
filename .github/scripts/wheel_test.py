from __future__ import annotations  # noqa: INP001

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from img2table.document.image import Image
from img2table.document.pdf import PDF


def assert_xlsx_buffer(buffer: BytesIO) -> None:
    buffer.seek(0)
    with ZipFile(buffer) as workbook:
        names = set(workbook.namelist())

    assert "[Content_Types].xml" in names
    assert "xl/workbook.xml" in names


def main() -> None:
    root = Path(__file__).resolve().parent

    image = Image(src=root / "test.bmp", detect_rotation=True)
    image_tables = image.extract_tables(
        borderless_tables=True, implicit_rows=True, min_confidence=50
    )
    assert image_tables
    assert_xlsx_buffer(
        image.to_xlsx(dest=BytesIO(), borderless_tables=True, implicit_rows=True, min_confidence=50)
    )

    pdf = PDF(src=root / "test.pdf", detect_rotation=True)
    pdf_tables = pdf.extract_tables(borderless_tables=True, implicit_rows=True, min_confidence=50)
    assert pdf_tables
    assert all(tables for tables in pdf_tables.values())
    assert_xlsx_buffer(
        pdf.to_xlsx(dest=BytesIO(), borderless_tables=True, implicit_rows=True, min_confidence=50)
    )


if __name__ == "__main__":
    main()
