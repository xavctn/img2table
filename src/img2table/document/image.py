# coding: utf-8
from typing import Iterator

import cv2
import numpy as np

from img2table.document import Document


class Image(Document):
    @property
    def images(self) -> Iterator[np.ndarray]:
        yield cv2.imdecode(np.frombuffer(self.bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
