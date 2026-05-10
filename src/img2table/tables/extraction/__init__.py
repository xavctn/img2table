from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from img2table.tables.extraction._utils import build_html, build_worksheet

if TYPE_CHECKING:
    from collections import OrderedDict

    import pandas as pd
    from xlsxwriter.format import Format
    from xlsxwriter.worksheet import Worksheet


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    image_width: int | None = field(default=None, repr=False, compare=False)
    image_height: int | None = field(default=None, repr=False, compare=False)

    @property
    def relative(self) -> RelativeBBox:
        if self.image_width is None or self.image_height is None:
            raise ValueError(
                "Relative bounding box is unavailable without source image dimensions."
            )

        return RelativeBBox(
            x1=self.x1 / self.image_width,
            y1=self.y1 / self.image_height,
            x2=self.x2 / self.image_width,
            y2=self.y2 / self.image_height,
        )


@dataclass
class RelativeBBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class TableCell:
    bbox: BBox
    value: str | None

    def __hash__(self) -> int:
        return hash((self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2, self.value))


@dataclass
class ExtractedTable:
    bbox: BBox
    title: str | None
    content: OrderedDict[int, list[TableCell]]

    @property
    def df(self) -> pd.DataFrame:
        """
        pandas DataFrame representation of the table
        """
        try:
            import pandas as pd
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Missing dependencies, please install 'pandas' to export table to dataframe."
            ) from err

        values = [[cell.value for cell in row] for row in self.content.values()]
        return pd.DataFrame(values)

    @property
    def html(self) -> str:
        """
        HTML representation of the table
        """
        return build_html(self)

    def _to_worksheet(self, sheet: Worksheet, cell_fmt: Format | None = None) -> None:
        """
        Populate xlsx worksheet with table data
        :param sheet: xlsxwriter Worksheet
        :param cell_fmt: xlsxwriter cell format
        """
        build_worksheet(table=self, sheet=sheet, cell_fmt=cell_fmt)

    def html_repr(self, title: str | None = None) -> str:
        """
        Create HTML representation of the table
        :param title: title of HTML paragraph
        :return: HTML string
        """
        html = f"""{rf'<h3 style="text-align: center">{title}</h3>' if title else ""}
                   <p style=\"text-align: center\">
                       <b>Title:</b> {self.title or "No title detected"}<br>
                       <b>Bounding box:</b> x1={self.bbox.x1}, y1={self.bbox.y1}, x2={self.bbox.x2}, y2={self.bbox.y2}
                   </p>
                   <div align=\"center\">{self.df.to_html().replace("None", "")}</div>
                   <hr>
                """
        return html  # noqa: RET504

    def __repr__(self) -> str:
        nb_columns = len(next(iter(self.content.values()), []))
        return (
            f"ExtractedTable(title={self.title}, bbox=({self.bbox.x1}, {self.bbox.y1}, {self.bbox.x2}, "
            f"{self.bbox.y2}),shape=({len(self.content)}, {nb_columns}))".strip()
        )
