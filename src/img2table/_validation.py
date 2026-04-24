import io
from pathlib import Path

import polars as pl


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


def validate_pages(pages: object) -> None:
    if pages is None:
        return

    if not isinstance(pages, list):
        msg = "pages must be a list[int] or None"
        raise ValidationError(msg)

    if any(not isinstance(page, int) or isinstance(page, bool) for page in pages):
        msg = "pages must contain only int values"
        raise ValidationError(msg)


def validate_polars_dataframe(df: object, field_name: str) -> None:
    if not isinstance(df, pl.DataFrame):
        msg = f"{field_name} must be a polars.DataFrame"
        raise ValidationError(msg)
