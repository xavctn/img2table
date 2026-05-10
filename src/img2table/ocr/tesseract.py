from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import cv2

from img2table.ocr._types import OCRData, OCRInstance

if TYPE_CHECKING:
    import numpy as np

    from img2table.document._types import Document, MockDocument


class TesseractOCR(OCRInstance):
    """
    Tesseract-OCR instance
    """

    def __init__(
        self,
        n_threads: int = 1,
        lang: str = "eng",
        psm: int = 11,
        tessdata_dir: str | None = None,
    ) -> None:
        """
        Initialization of Tesseract OCR instance
        :param n_threads: number of concurrent threads used for Tesseract
        :param lang: lang parameter used in Tesseract
        :param psm: PSM parameter used in Tesseract
        :param tessdata_dir: directory containing Tesseract traineddata files
        """
        if isinstance(n_threads, int):
            self.n_threads = n_threads
        else:
            raise TypeError(f"Invalid type {type(n_threads)} for n_threads argument")

        if isinstance(lang, str):
            self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        if isinstance(psm, int):
            self.psm = psm
        else:
            raise TypeError(f"Invalid type {type(psm)} for psm argument")

        # Create custom environment
        env = os.environ.copy()
        if tessdata_dir:
            env["TESSDATA_PREFIX"] = tessdata_dir
        self.env = env

        # Check if Tesseract is available
        cmd_tess = subprocess.run("tesseract --version", env=self.env, shell=True, check=False)  # noqa: S602, S607
        if cmd_tess.returncode != 0:
            raise OSError("Tesseract not found in environment. Check variables and PATH")

        # Check if requested languages are available
        try:
            lang_tess = subprocess.check_output(  # noqa: S602
                "tesseract --list-langs",  # noqa: S607
                env=self.env,
                shell=True,
            ).decode()
            for lng in self.lang.split("+"):
                if not any(
                    re.search(rf"\b{lng}\b", line) is not None for line in lang_tess.splitlines()
                ):
                    raise OSError(f"Tesseract '{lng}' trainned data cannot be located")
        except subprocess.CalledProcessError as err:
            raise OSError("Tesseract trainned data cannot be located.") from err

    def hocr(self, image: np.ndarray) -> str:
        """
        Get hOCR HTML of an image using Tesseract
        :param image: numpy array representing the image
        :return: hOCR HTML string
        """
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_f:
            tmp_file = tmp_f.name
            # Write image to temporary file
            cv2.imwrite(tmp_file, image)

            # Get hOCR
            hocr = subprocess.check_output(  # noqa: S602
                f"tesseract {tmp_file} stdout --psm {self.psm} -l {self.lang} hocr",
                env=self.env,
                stderr=subprocess.STDOUT,
                shell=True,
            )

        # Remove temporary file
        while Path(tmp_file).exists():
            with suppress(PermissionError):
                Path(tmp_file).unlink(missing_ok=True)

        return hocr.decode("utf-8")

    def of(self, document: Document | MockDocument) -> OCRData | None:
        """
        Convert hOCR HTML to OCRData object
        :param content: hOCR HTML string
        :return: OCRData object corresponding to content
        """
        # Apply OCR on all pages
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            content = pool.map(self.hocr, document.images)

        from bs4 import BeautifulSoup

        # Create dict of OCR elements by page
        records = {}

        for page, hocr in enumerate(content):
            # Instantiate HTML parser
            soup = BeautifulSoup(hocr, features="html.parser")

            # Parse all HTML elements
            list_elements = []
            for element in soup.find_all(class_=True):
                if element["class"][0] != "ocrx_word":
                    continue

                # Parse properties
                str_conf = (
                    re.findall(r"x_wconf \d{1,2}", title)
                    if isinstance((title := element["title"]), str)
                    else []
                )
                confidence = int(str_conf[0].split()[1]) if str_conf else None

                bbox = re.findall(r"bbox \d{1,4} \d{1,4} \d{1,4} \d{1,4}", element["title"])[0]  # ty:ignore[no-matching-overload]
                x1, y1, x2, y2 = tuple(map(int, re.sub(r"^bbox\s", "", bbox).split()))

                # Get element properties
                list_elements.append(
                    {
                        "id": element["id"],
                        "parent": element.parent.get("id") if element.parent else None,
                        "value": re.sub(r"^(\s|\||L|_|;|\*)*$", "", element.string).strip() or None
                        if element.string
                        else None,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

            # Create dataframe
            if list_elements:
                records[page] = list_elements

        return OCRData(records=records) if records else None
