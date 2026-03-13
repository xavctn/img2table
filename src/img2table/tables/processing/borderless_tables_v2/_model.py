from dataclasses import dataclass, field

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
    _ws_cache: dict[tuple[float, int], list[Whitespace]] = field(default_factory=dict)

    def add(self, item: Cell) -> None:
        self.items.append(item)

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2

    def compute_whitespaces(self, min_width: float, width: int) -> list[Whitespace]:
        """
        Compute whitespaces between cells in a merged row.
        :param min_width: minimum width for a whitespace to be considered
        :param width: total width of the row
        :return: list of whitespaces
        """
        cache_key = (min_width, width)
        if cache_key in self._ws_cache:
            return self._ws_cache[cache_key]

        # Compute whitespaces
        self._ws_cache[cache_key] = compute_whitespaces(
            items=self.items, min_width=min_width, width=width
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


def compute_whitespaces(items: list[Cell], min_width: float, width: int) -> list[Whitespace]:
    """
    Compute whitespaces between cells.
    :param items: list of cells
    :param min_width: minimum width for a whitespace to be considered
    :param width: total width of the row
    :return: list of whitespaces
    """
    current_x, whitespaces = 0, []
    for item in sorted(items, key=lambda it: it.x1):
        # Check gap width
        gap = item.x1 - current_x
        if current_x == 0 or gap >= min_width:
            whitespaces.append(Whitespace(start=current_x, end=item.x1, start_bound=current_x == 0))
        current_x = max(current_x, item.x2)

    # Add last whitespace
    whitespaces.append(Whitespace(start=current_x, end=width, end_bound=True))

    return whitespaces
