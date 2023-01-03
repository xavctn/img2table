# coding: utf-8
import io
from dataclasses import dataclass
from typing import Union, Iterator

import numpy as np

from img2table.ocr.base import OCRInstance


@dataclass
class Document:
    src: Union[str, io.BytesIO, bytes]
    dpi: int = 300
    ocr: "OCRInstance" = None

    @property
    def bytes(self) -> bytes:
        if isinstance(self.src, bytes):
            return self.src
        elif isinstance(self.src, io.BytesIO):
            return self.src.read()
        elif isinstance(self.src, str):
            with io.open(self.src, 'rb') as f:
                return f.read()

    @property
    def images(self) -> Iterator[np.ndarray]:
        raise NotImplementedError
