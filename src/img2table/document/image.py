# coding: utf-8
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from img2table.document.base import Document


@dataclass
class Image(Document):
    dpi: int = 200

    def __post_init__(self):
        self.pages = None
        super(Image, self).__post_init__()

    @property
    def images(self) -> Iterator[np.ndarray]:
        yield cv2.imdecode(np.frombuffer(self.bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
