import csv
from pathlib import Path

from img2table.tables.types import Cell


def read_cells(path: str) -> list[Cell]:
    with Path(path).open(encoding="utf-8") as f:
        return [
            Cell(x1=int(row["x1"]), x2=int(row["x2"]), y1=int(row["y1"]), y2=int(row["y2"]))
            for row in csv.DictReader(f, delimiter=";")
        ]
