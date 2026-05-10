from pathlib import Path

import cv2
import pytest

from img2table.document.image import Image
from img2table.ocr import TesseractOCR
from tests import MOCK_DIR, TESSERACT_INSTALL
from tests.ocr_data_utils import read_ocr_data


def test_validators() -> None:
    with pytest.raises(TypeError):
        TesseractOCR(n_threads=[8])  # ty:ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        TesseractOCR(lang=12)  # ty:ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        TesseractOCR(psm="r")  # ty:ignore[invalid-argument-type]


@pytest.mark.skipif(TESSERACT_INSTALL, reason="Tesseract installed locally")
def test_installed() -> None:
    with pytest.raises(EnvironmentError):
        TesseractOCR()


def test_lang_validators(mock_tesseract) -> None:  # noqa: ANN001, ARG001
    with pytest.raises(EnvironmentError):
        TesseractOCR(lang="zzz")


def test_tesseract_hocr(mock_tesseract) -> None:  # noqa: ANN001, ARG001
    instance = TesseractOCR()
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = instance.hocr(image=img)

    with (Path(MOCK_DIR) / "tesseract_hocr.html").open() as f:
        assert result == f.read()


def test_tesseract_document(mock_tesseract) -> None:  # noqa: ANN001, ARG001
    instance = TesseractOCR()
    doc = Image(src="test_data/test.png")

    result = instance.of(document=doc)

    expected = read_ocr_data("test_data/ocr.csv")

    assert result == expected
