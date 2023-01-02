# coding: utf-8
from dataclasses import dataclass
from enum import Enum, auto
from io import BytesIO
from typing import Union, Iterator

import numpy as np


class DocumentType(Enum):
    PDF = auto()
    IMAGE = auto()


@dataclass
class Document:
    src: Union[str, BytesIO, bytes]
    typ: DocumentType
    dpi: int = 300

    @property
    def images(self) -> Iterator[np.ndarray]:
        raise NotImplementedError
