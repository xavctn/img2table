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


def assert_ocr_content_close(
    actual: OCRData, expected: OCRData, tolerance: int = 5, check_ids: bool = False
) -> None:
    """Compare recognized text and boxes while allowing small engine coordinate changes."""
    assert actual.records.keys() == expected.records.keys()
    for page, expected_records in expected.records.items():
        actual_records = actual.records[page]
        assert [rec["value"] for rec in actual_records] == [rec["value"] for rec in expected_records]
        if check_ids:
            assert [(rec["id"], rec["parent"]) for rec in actual_records] == [
                (rec["id"], rec["parent"]) for rec in expected_records
            ]
        for actual_rec, expected_rec in zip(actual_records, expected_records, strict=True):
            for field in ("x1", "y1", "x2", "y2"):
                assert abs(actual_rec[field] - expected_rec[field]) <= tolerance, (
                    page, actual_rec["value"], field, actual_rec[field], expected_rec[field]
                )
