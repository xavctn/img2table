# coding: utf-8
from dataclasses import dataclass
from io import BytesIO
from typing import Union, Iterator

import numpy as np


@dataclass
class Document:
    src: Union[str, BytesIO, bytes]
    dpi: int = 300

    @property
    def images(self) -> Iterator[np.ndarray]:
        raise NotImplementedError
