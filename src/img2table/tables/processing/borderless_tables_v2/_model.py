from dataclasses import dataclass, field

from img2table.tables.objects import TableObject
from img2table.tables.objects.cell import Cell


@dataclass
class ItemHolder:
    items: list[Cell] = field(default_factory=list, repr=False)

    @property
    def x1(self) -> int:
        return min((it.x1 for it in self.items), default=0)

    @property
    def y1(self) -> int:
        return min((it.y1 for it in self.items), default=0)

    @property
    def x2(self) -> int:
        return max((it.x2 for it in self.items), default=0)

    @property
    def y2(self) -> int:
        return max((it.y2 for it in self.items), default=0)

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Whitespace:
    start: int
    end: int
    start_bound: bool = False
    end_bound: bool = False

    @property
    def width(self) -> int:
        return self.end - self.start

    def matching_bound(self, other: "Whitespace") -> bool:
        if self.start_bound and other.start_bound:
            return True
        return self.end_bound and other.end_bound

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.start_bound, self.end_bound))


@dataclass
class MergedRow(ItemHolder):
    _ws_cache: dict[tuple[float, int, int], list[Whitespace]] = field(default_factory=dict)

    def add(self, item: Cell) -> None:
        self.items.append(item)

    def compute_whitespaces(self, min_width: float, x_min: int, x_max: int) -> list[Whitespace]:
        """
        Compute whitespaces between cells in a merged row.
        :param min_width: minimum width for a whitespace to be considered
        :param x_min: start of span
        :param x_max: end of span
        :return: list of whitespaces
        """
        cache_key = (min_width, x_min, x_max)
        if cache_key in self._ws_cache:
            return self._ws_cache[cache_key]

        # Compute whitespaces
        self._ws_cache[cache_key] = compute_whitespaces(
            items=self.items, min_width=min_width, x_min=x_min, x_max=x_max
        )
        return self._ws_cache[cache_key]


@dataclass
class ColumnSection(ItemHolder):
    rows: list[MergedRow] = field(default_factory=list, repr=False)
    whitespaces: list[Whitespace] = field(default_factory=list)

    def update(self, row: MergedRow, whitespaces: list[Whitespace]) -> "ColumnSection":
        self.items += row.items
        self.rows += [row]
        self.whitespaces = whitespaces

        return self

    @property
    def nb_columns(self) -> int:
        return max(0, len(self.whitespaces) - 1)

    @property
    def first_y_center(self) -> float:
        return min((row.y_center for row in self.rows), default=0)

    @property
    def last_y_center(self) -> float:
        return max((row.y_center for row in self.rows), default=0)


@dataclass
class LayoutRegion(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int
    contours: list[Cell] = field(default_factory=list, repr=False)

    @classmethod
    def build(cls, x1: int, y1: int, x2: int, y2: int, contours: list[Cell]) -> "LayoutRegion":
        return cls(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            contours=[
                cnt
                for cnt in contours
                if (y_overlap := max(0, min(cnt.y2, y2) - max(cnt.y1, y1))) > 0
                and (x_overlap := max(0, min(cnt.x2, x2) - max(cnt.x1, x1))) > 0
                and x_overlap * y_overlap >= 0.5 * cnt.area
            ],
        )


@dataclass
class RowCharacteristic:
    index: int
    row: MergedRow
    ws: list[Whitespace]

    @property
    def count(self) -> int:
        return len(self.ws)

    @property
    def inner_ws_width(self) -> int:
        inner_ws = [ws for ws in self.ws if not ws.start_bound and not ws.end_bound]
        return sum(ws.width for ws in inner_ws) if inner_ws else 0


def compute_whitespaces(
    items: list[Cell], min_width: float, x_min: int, x_max: int
) -> list[Whitespace]:
    """
    Compute whitespaces between cells.
    :param items: list of cells
    :param min_width: minimum width for a whitespace to be considered
    :param x_min: start of span
    :param x_max: end of span
    :return: list of whitespaces
    """
    current_x, whitespaces = x_min, []
    for item in sorted(items, key=lambda it: it.x1):
        # Check gap width
        gap = item.x1 - current_x
        if current_x == x_min or gap >= min_width:
            whitespaces.append(
                Whitespace(start=current_x, end=item.x1, start_bound=current_x == x_min)
            )
        current_x = max(current_x, item.x2)

    # Add last whitespace
    whitespaces.append(Whitespace(start=current_x, end=x_max, end_bound=True))

    return whitespaces


def identify_merged_rows(cnts: list[Cell]) -> list[MergedRow]:
    """
    Identify merged rows in a list of cells.
    :param cnts: list of cells
    :return: list of merged rows
    """
    if not cnts:
        return []

    current_row, merged_rows = None, []
    for cnt in sorted(cnts, key=lambda cnt: (cnt.y1, cnt.x1)):
        if current_row is None:
            current_row = MergedRow(items=[cnt])
            continue

        # Compute overlap
        overlap = min(current_row.y2, cnt.y2) - max(current_row.y1, cnt.y1)
        if overlap <= 0.5 * min(cnt.height, current_row.height):
            # Flush current row
            merged_rows.append(current_row)
            current_row = MergedRow()
        current_row.add(cnt)

    # Add last row
    merged_rows.append(current_row)

    return merged_rows
