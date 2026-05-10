from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from img2table.ocr._types import OCRData

INT_FIELDS = {"page", "x1", "y1", "x2", "y2"}


def _coerce_csv_value(field: str, value: str) -> Any:
    if value == "":
        return None
    if field in INT_FIELDS:
        return int(value)
    if field == "confidence":
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def read_ocr_data(path: str | Path) -> OCRData:
    with Path(path).open(encoding="utf-8", newline="") as f:
        records: dict[int, list[dict[str, Any]]] = {}
        for row in csv.DictReader(f, delimiter=";"):
            record = {
                field: _coerce_csv_value(field=field, value=value) for field, value in row.items()
            }
            record_page = record.pop("page")
            records.setdefault(record_page, []).append(record)
    return OCRData(records=records)


def drop_record_fields(
    records: dict[int, list[dict[str, Any]]], fields: set[str]
) -> dict[int, list[dict[str, Any]]]:
    return {
        page: [
            {key: value for key, value in rec.items() if key not in fields} for rec in page_records
        ]
        for page, page_records in records.items()
    }
