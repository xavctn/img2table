"""Generate the table-detection figures embedded in the algorithm overview."""  # noqa: INP001

from pathlib import Path

import cv2
import numpy as np

from img2table.tables.borderless import extract_borderless_tables
from img2table.tables.borderless.layout import identify_image_layout
from img2table.tables.borderless.layout.text_lines import identify_text_mask
from img2table.tables.extractor import TableExtractor
from img2table.tables.types import Table

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "docs" / "images" / "table-detection"


def load_image(path: Path) -> np.ndarray:
    """Load an image in the RGB representation expected by the extractor."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_lines(image: np.ndarray, extractor: TableExtractor) -> np.ndarray:
    """Draw detected horizontal and vertical lines over an image."""
    result = image.copy()
    for line in extractor.horizontal_lines:
        cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), (220, 40, 40), 2)
    for line in extractor.vertical_lines:
        cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), (40, 100, 220), 2)
    return result


def draw_tables(image: np.ndarray, tables: list[Table]) -> np.ndarray:
    """Draw reconstructed table cells over an image."""
    result = image.copy()
    for table in tables:
        for row in table.rows:
            for cell in row.cells:
                cv2.rectangle(result, (cell.x1, cell.y1), (cell.x2, cell.y2), (35, 150, 60), 2)
    return result


def make_figure(panels: list[tuple[str, np.ndarray]], output: Path) -> None:
    """Lay out labelled image panels in a two-column PNG."""
    panel_width, panel_height, label_height, gap = 720, 360, 42, 24
    canvas = np.full(
        (2 * (panel_height + label_height) + gap, 2 * panel_width + gap, 3), 255, np.uint8
    )

    for index, (label, image) in enumerate(panels):
        scale = min(panel_width / image.shape[1], panel_height / image.shape[0])
        size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        row, col = divmod(index, 2)
        x = col * (panel_width + gap) + (panel_width - size[0]) // 2
        y = row * (panel_height + label_height + gap) + label_height + (panel_height - size[1]) // 2
        cv2.putText(
            canvas,
            label,
            (col * (panel_width + gap), row * (panel_height + label_height + gap) + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        canvas[y : y + size[1], x : x + size[0]] = resized

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def bordered_figure() -> None:
    """Render foreground, lines, and reconstruction for a ruled-table example."""
    image = load_image(ROOT / "examples" / "data" / "tables.png")
    extractor = TableExtractor(img=image)
    extractor.extract_bordered_tables()
    make_figure(
        [
            ("1. Input image", image),
            ("2. Foreground mask", cv2.cvtColor(extractor.thresh, cv2.COLOR_GRAY2RGB)),
            ("3. Horizontal (red) and vertical (blue) lines", draw_lines(image, extractor)),
            ("4. Reconstructed cell grid", draw_tables(image, extractor.tables)),
        ],
        OUTPUT_DIR / "bordered-pipeline.png",
    )


def borderless_figure() -> None:
    """Render text layout and reconstruction for a borderless-table example."""
    image = load_image(ROOT / "examples" / "data" / "borderless" / "2.png")
    extractor = TableExtractor(img=image)
    if extractor.characteristics is None:
        raise RuntimeError("Could not compute image characteristics")
    extractor.extract_bordered_tables()
    text_mask = identify_text_mask(
        thresh=extractor.thresh.copy(),
        lines=extractor.lines,
        char_length=extractor.characteristics.char_length,
        existing_tables=extractor.tables,
    )
    layout = identify_image_layout(
        thresh=extractor.thresh.copy(),
        lines=extractor.lines,
        char_length=extractor.characteristics.char_length,
        existing_tables=extractor.tables,
    )
    layout_overlay = image.copy()
    for region in layout:
        cv2.rectangle(
            layout_overlay, (region.x1, region.y1), (region.x2, region.y2), (160, 60, 180), 2
        )
    tables = extract_borderless_tables(
        thresh=extractor.thresh.copy(),
        lines=extractor.lines,
        char_length=extractor.characteristics.char_length,
        existing_tables=extractor.tables,
    )
    make_figure(
        [
            ("1. Input image", image),
            ("2. Text-layout mask", cv2.cvtColor(text_mask, cv2.COLOR_GRAY2RGB)),
            ("3. Layout regions", layout_overlay),
            ("4. Reconstructed borderless grid", draw_tables(image, tables)),
        ],
        OUTPUT_DIR / "borderless-pipeline.png",
    )


if __name__ == "__main__":
    bordered_figure()
    borderless_figure()
