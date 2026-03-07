from dataclasses import dataclass, field

from img2table.tables.objects.cell import Cell


@dataclass
class ItemHolder:
    items: list[Cell] = field(default_factory=list)

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
class MergedRow(ItemHolder):
    def add(self, item: Cell) -> None:
        self.items.append(item)


@dataclass
class Whitespace:
    start: int
    end: int
    start_bound: bool = False
    end_bound: bool = False

    def matching_bound(self, other: "Whitespace") -> bool:
        if self.start_bound and other.start_bound:
            return True
        return self.end_bound and other.end_bound

    def __hash__(self) -> int:
        return hash(repr(self))


@dataclass
class ColumnSection(ItemHolder):
    whitespaces: list[Whitespace] = field(default_factory=list)

    def update(self, row: MergedRow, whitespaces: list[Whitespace]) -> None:
        self.items += row.items
        self.whitespaces = whitespaces

    @property
    def nb_columns(self) -> int:
        return max(0, len(self.whitespaces) - 1)


def compute_whitespaces(row: MergedRow, min_width: float, width: int) -> list[Whitespace]:
    """
    Compute whitespaces between cells in a merged row.
    :param row: merged row to compute whitespaces for
    :param min_width: minimum width for a whitespace to be considered
    :param width: total width of the row
    :return: list of whitespaces
    """
    current_x, whitespaces = 0, []
    for item in sorted(row.items, key=lambda it: it.x1):
        # Check gap width
        gap = item.x1 - current_x
        if current_x == 0 or gap >= min_width:
            whitespaces.append(Whitespace(start=current_x, end=item.x1, start_bound=current_x == 0))
        current_x = max(current_x, item.x2)

    # Add last whitespace
    whitespaces.append(Whitespace(start=current_x, end=width, end_bound=True))

    return whitespaces


def matching_whitespaces(
    ws1_list: list[Whitespace], ws2_list: list[Whitespace], min_width: float
) -> tuple[bool, list[Whitespace]]:
    """
    Identify if both sets of whitespaces match
    :param ws1_list: first set of whitespaces
    :param ws2_list: second set of whitespaces
    :param min_width: minimum column width
    :return: boolean indicating whether two sets of whitespaces match and resultant whitespaces
    """
    # Get largest and smallest list of whitespaces and iterate over the shortest list
    ws_short, ws_long = (
        (ws1_list, ws2_list) if len(ws1_list) <= len(ws2_list) else (ws2_list, ws1_list)
    )

    matching_ws, covered_long_indices = [], set()
    for ws_s in ws_short:
        found_matching_ws = False
        for idx, ws_l in enumerate(ws_long):
            # Compute overlap
            overlap = min(ws_s.end, ws_l.end) - max(ws_s.start, ws_l.start)

            # Check overlap is sufficient or if bounds match
            if overlap >= min_width or ws_s.matching_bound(ws_l):
                matching_ws.append(
                    Whitespace(
                        start=max(ws_s.start, ws_l.start),
                        end=min(ws_s.end, ws_l.end),
                        start_bound=ws_s.start_bound and ws_l.start_bound,
                        end_bound=ws_s.end_bound and ws_l.end_bound,
                    )
                )
                covered_long_indices.add(idx)
                found_matching_ws = True

        if not found_matching_ws:
            return False, []

    if len(covered_long_indices) < len(ws_long):
        return False, []

    return True, sorted(matching_ws, key=lambda x: x.start)


def identify_merged_rows(cnts: list[Cell]) -> list[MergedRow]:
    """
    Identify merged rows in a list of cells.
    :param cnts: list of cells
    :return: list of merged rows
    """
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


def compute_column_section(
    merged_rows: list[MergedRow], min_width: float, width: int
) -> list[ColumnSection]:
    """
    Compute column sections from merged rows.
    :param merged_rows: list of merged rows
    :param min_width: minimum width for a whitespace to be considered
    :param width: total width of the row
    :return: list of column sections
    """
    column_sections: list[ColumnSection] = []
    current_section = ColumnSection()
    for idx_row, row in enumerate(merged_rows):
        row_ws = compute_whitespaces(row=row, min_width=min_width, width=width)

        # First iteration
        if idx_row == 0:
            current_section.update(row=row, whitespaces=row_ws)
            continue
        # Going from no column to multi-column
        if current_section.nb_columns == 1 and len(row_ws) > 2:
            # Flush current section and create new one
            column_sections.append(current_section)
            current_section = ColumnSection(items=row.items, whitespaces=row_ws)
            continue

        # Compute correspondence between row whitespaces and current whitespaces
        is_matching, matching_ws = matching_whitespaces(
            ws1_list=current_section.whitespaces, ws2_list=row_ws, min_width=min_width
        )

        if is_matching:
            # Update current section
            current_section.update(row=row, whitespaces=matching_ws)
        else:
            # Flush current section and create new one
            column_sections.append(current_section)
            current_section = ColumnSection(items=row.items, whitespaces=row_ws)

    # Flush current section
    column_sections.append(current_section)

    return column_sections
