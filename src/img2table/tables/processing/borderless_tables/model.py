from dataclasses import dataclass, field

from img2table.tables.objects.cell import Cell


@dataclass
class Whitespace:
    cells: list[Cell]

    @property
    def x1(self) -> int:
        return min([c.x1 for c in self.cells])

    @property
    def y1(self) -> int:
        return min([c.y1 for c in self.cells])

    @property
    def x2(self) -> int:
        return max([c.x2 for c in self.cells])

    @property
    def y2(self) -> int:
        return max([c.y2 for c in self.cells])

    @property
    def width(self) -> int:
        return sum([c.width for c in self.cells])

    @property
    def height(self) -> int:
        return sum([c.height for c in self.cells])

    @property
    def area(self) -> int:
        return sum([c.area for c in self.cells])

    @property
    def continuous(self) -> bool:
        return len(self.cells) == 1

    def flipped(self) -> "Whitespace":
        return Whitespace(cells=[Cell(x1=c.y1, y1=c.x1, x2=c.y2, y2=c.x2) for c in self.cells])

    def __contains__(self, item: "Whitespace") -> bool:
        return self.x1 <= item.x1 and self.y1 <= item.y1 and self.x2 >= item.x2 and self.y2 >= item.y2

    def __hash__(self) -> int:
        return hash(repr(self))


@dataclass
class ImageSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    elements: list[Cell] = None
    whitespaces: list[Whitespace] = None
    position: int = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def element_height(self) -> int:
        if self.elements:
            return max([el.y2 for el in self.elements]) - min([el.y1 for el in self.elements])
        return self.height

    def set_elements(self, elements: list[Cell]) -> None:
        self.elements = elements

    def set_whitespaces(self, whitespaces: list[Whitespace]) -> None:
        self.whitespaces = whitespaces

    def __hash__(self) -> int:
        return hash(repr(self))


@dataclass
class TableSegment:
    table_areas: list[ImageSegment]

    @property
    def x1(self) -> int:
        return min([tb_area.x1 for tb_area in self.table_areas])

    @property
    def y1(self) -> int:
        return min([tb_area.y1 for tb_area in self.table_areas])

    @property
    def x2(self) -> int:
        return max([tb_area.x2 for tb_area in self.table_areas])

    @property
    def y2(self) -> int:
        return max([tb_area.y2 for tb_area in self.table_areas])

    @property
    def elements(self) -> list[Cell]:
        return [el for tb_area in self.table_areas for el in tb_area.elements]

    @property
    def whitespaces(self) -> list[Whitespace]:
        return [ws for tb_area in self.table_areas for ws in tb_area.whitespaces]


@dataclass
class VerticalWS:
    ws: Whitespace
    position: int = 0
    top: bool = True
    bottom: bool = True
    used: bool = False

    @property
    def x1(self) -> int:
        return self.ws.x1

    @property
    def y1(self) -> int:
        return self.ws.y1

    @property
    def x2(self) -> int:
        return self.ws.x2

    @property
    def y2(self) -> int:
        return self.ws.y2

    @property
    def width(self) -> int:
        return self.ws.x2 - self.ws.x1

    @property
    def height(self) -> int:
        return self.ws.y2 - self.ws.y1

    @property
    def continuous(self) -> bool:
        return self.ws.continuous


@dataclass
class Column:
    whitespaces: list[VerticalWS]
    top: bool = True
    bottom: bool = True
    top_position: int = 0
    bottom_position: int = 0

    @property
    def x1(self) -> int:
        return max([v_ws.ws.x1 for v_ws in self.whitespaces])

    @property
    def y1(self) -> int:
        return min([v_ws.ws.y1 for v_ws in self.whitespaces])

    @property
    def x2(self) -> int:
        return min([v_ws.ws.x2 for v_ws in self.whitespaces])

    @property
    def y2(self) -> int:
        return max([v_ws.ws.y2 for v_ws in self.whitespaces])

    @property
    def height(self) -> int:
        y_values = {y for v_ws in self.whitespaces for c in v_ws.ws.cells for y in range(c.y1, c.y2 + 1)}
        return len(y_values) - 1

    @property
    def continuous(self) -> bool:
        return all(v_ws.continuous for v_ws in self.whitespaces)

    @classmethod
    def from_ws(cls, v_ws: VerticalWS) -> "Column":
        return cls(whitespaces=[v_ws], top=v_ws.top, bottom=v_ws.bottom, top_position=v_ws.position,
                   bottom_position=v_ws.position)

    def corresponds(self, v_ws: VerticalWS, char_length: float) -> bool:
        if self.bottom_position is None:
            return True
        if v_ws.position != self.bottom_position + 1:
            return False
        if not self.bottom or not v_ws.top:
            return False

        # Condition on position
        return min(self.x2, v_ws.x2) - max(self.x1, v_ws.x1) >= 0.5 * char_length

    def add(self, v_ws: VerticalWS) -> None:
        self.whitespaces.append(v_ws)
        self.top_position = min(self.top_position, v_ws.position)
        self.bottom_position = max(self.bottom_position, v_ws.position)

        if v_ws.position == self.top_position:
            self.top = v_ws.top

        if v_ws.position == self.bottom_position:
            self.bottom = v_ws.bottom


@dataclass
class ColumnGroup:
    columns: list[Column]
    char_length: float
    elements: list[Cell] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Reprocess left and right columns positions
        self.columns = sorted(self.columns, key=lambda col: col.x1)

        if len(self.columns) >= 2 and len(self.elements) > 0:
            x_left, x_right = min([el.x1 for el in self.elements]), max([el.x2 for el in self.elements])
            # Left column
            self.columns[0] = Column(whitespaces=[
                VerticalWS(ws=Whitespace(cells=[Cell(x1=x_left - int(0.5 * self.char_length),
                                                     y1=c.y1,
                                                     x2=x_left - int(0.5 * self.char_length),
                                                     y2=c.y2)
                                                for c in v_ws.ws.cells]))
                for v_ws in self.columns[0].whitespaces
            ])

            # Right column
            self.columns[-1] = Column(whitespaces=[
                VerticalWS(ws=Whitespace(cells=[Cell(x1=x_right + int(0.5 * self.char_length),
                                                     y1=c.y1,
                                                     x2=x_right + int(0.5 * self.char_length),
                                                     y2=c.y2)
                                                for c in v_ws.ws.cells]))
                for v_ws in self.columns[-1].whitespaces
            ])

    @property
    def x1(self) -> int:
        if self.columns:
            return min([d.x1 for d in self.columns])
        return 0

    @property
    def y1(self) -> int:
        if self.columns:
            return min([d.y1 for d in self.columns])
        return 0

    @property
    def x2(self) -> int:
        if self.columns:
            return max([d.x2 for d in self.columns])
        return 0

    @property
    def y2(self) -> int:
        if self.columns:
            return max([d.y2 for d in self.columns])
        return 0

    @property
    def bbox(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ColumnGroup):
            try:
                assert self.columns == other.columns
                assert set(self.elements) == set(other.elements)
                return True
            except AssertionError:
                return False
        return False
