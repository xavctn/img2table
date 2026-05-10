import io
from pathlib import Path
from typing import Any, cast


class ValidationError(ValueError):
    pass


def validate_src(src: object) -> None:
    if not isinstance(src, (str, Path, io.BytesIO, bytes)):
        msg = "src must be a str, Path, BytesIO, or bytes"
        raise ValidationError(msg)


def validate_bool(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        msg = f"{field_name} must be a bool"
        raise ValidationError(msg)


def validate_positive_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{field_name} must be a positive int"
        raise ValidationError(msg)


def validate_pages(pages: object) -> None:
    if pages is None:
        return

    if not isinstance(pages, list):
        msg = "pages must be a list[int] or None"
        raise ValidationError(msg)

    if any(not isinstance(page, int) or isinstance(page, bool) for page in pages):
        msg = "pages must contain only int values"
        raise ValidationError(msg)


def validate_ocr_records(records: object, field_name: str) -> None:
    required_fields = {
        "id",
        "parent",
        "value",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    if not isinstance(records, dict) or len(records) == 0:
        msg = f"{field_name} must be a dict[int, list[dict]]"
        raise ValidationError(msg)

    for page, page_records in records.items():
        if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or not isinstance(page_records, list)
            or any(not isinstance(rec, dict) for rec in page_records)
        ):
            msg = f"{field_name} must be a dict[int, list[dict]]"
            raise ValidationError(msg)

        typed_records = cast("list[dict[str, Any]]", page_records)
        if any(not required_fields.issubset(rec) for rec in typed_records):
            msg = f"{field_name} records must contain OCR fields"
            raise ValidationError(msg)
