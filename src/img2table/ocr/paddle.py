
import warnings
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional

import cv2
import numpy as np
import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe


class PaddleOCR2(OCRInstance):
    """
    Paddle-OCR 2.X instance
    """
    def __init__(self, lang: str = 'en', kw: Optional[dict] = None) -> None:
        """
        Initialization of Paddle OCR instance
        :param lang: lang parameter used in Paddle
        :param kw: dictionary containing kwargs for PaddleOCR constructor
        """
        if isinstance(lang, str):
            self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        # Create kwargs dict for constructor
        kw = kw or {}
        kw["lang"] = self.lang
        kw["use_angle_cls"] = kw.get("use_angle_cls") or False
        kw["show_log"] = kw.get("show_log") or False

        from paddleocr import PaddleOCR as OCR
        self.ocr = OCR(**kw)

    def hocr(self, image: np.ndarray) -> list:
        """
        Get OCR of an image using Paddle
        :param image: numpy array representing the image
        :return: Paddle OCR result
        """
        with NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_f:
            tmp_file = tmp_f.name
            # Write image to temporary file
            cv2.imwrite(tmp_file, image)

            # Get OCR
            ocr_result = self.ocr.ocr(img=tmp_file, cls=False)

        # Remove temporary file
        while Path(tmp_file).exists():
            try:
                Path(tmp_file).unlink(missing_ok=True)
            except PermissionError:
                pass

        # Get result
        ocr_result = ocr_result.pop()
        return [[bbox, (word[0], round(word[1], 2))] for bbox, word in ocr_result] if ocr_result else []

    def content(self, document: Document) -> list[list]:
        # Get OCR of all images
        return [self.hocr(image=image) for image in document.images]

    def to_ocr_dataframe(self, content: list[list]) -> OCRDataframe:
        """
        Convert hOCR HTML to OCRDataframe object
        :param content: hOCR HTML string
        :return: OCRDataframe object corresponding to content
        """
        # Create list of elements
        list_elements = []

        for page, ocr_result in enumerate(content):
            for idx, (bbox, word) in enumerate(ocr_result):
                dict_word = {
                    "page": page,
                    "class": "ocrx_word",
                    "id": f"word_{page + 1}_{idx + 1}",
                    "parent": f"word_{page + 1}_{idx + 1}",
                    "value": word[0],
                    "confidence": 100 * word[1],
                    "x1": round(min([edge[0] for edge in bbox])),
                    "y1": round(min([edge[1] for edge in bbox])),
                    "x2": round(max([edge[0] for edge in bbox])),
                    "y2": round(max([edge[1] for edge in bbox]))
                }

                list_elements.append(dict_word)

        return OCRDataframe(df=pl.DataFrame(list_elements, schema=self.pl_schema)) if list_elements else None


class PaddleOCR3(OCRInstance):
    """
    Paddle-OCR 3.X instance
    """

    def __init__(self, lang: str = 'en', kw: Optional[dict] = None) -> None:
        """
        Initialization of Paddle OCR instance
        :param lang: lang parameter used in Paddle
        :param kw: dictionary containing kwargs for PaddleOCR constructor
        """
        if isinstance(lang, str):
            self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        # Create kwargs dict for constructor
        kw = kw or {}
        kw["lang"] = self.lang
        kw["use_doc_unwarping"] = kw.get("use_doc_unwarping") or False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from paddleocr import PaddleOCR as OCR

        self.ocr = OCR(**kw)

    def content(self, document: Document) -> list[dict]:
        ocrs = self.ocr.predict(input=document.images)
        return [{"rec_texts": res["rec_texts"],
                 "rec_scores": res["rec_scores"],
                 "rec_boxes": [bbox.tolist() for bbox in res["rec_boxes"]]}
                for res in ocrs]

    def to_ocr_dataframe(self, content: list[dict]) -> OCRDataframe:
        """
        Convert hOCR HTML to OCRDataframe object
        :param content: hOCR HTML string
        :return: OCRDataframe object corresponding to content
        """
        # Create list of elements
        list_elements = []

        for page, ocr_result in enumerate(content):
            for idx, (word, conf, bbox) in enumerate(zip(ocr_result["rec_texts"], ocr_result["rec_scores"], ocr_result["rec_boxes"])):
                dict_word = {
                    "page": page,
                    "class": "ocrx_word",
                    "id": f"word_{page + 1}_{idx + 1}",
                    "parent": f"word_{page + 1}_{idx + 1}",
                    "value": word,
                    "confidence": 100 * conf,
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3])
                }

                list_elements.append(dict_word)

        return OCRDataframe(df=pl.DataFrame(list_elements, schema=self.pl_schema)) if list_elements else None


class PaddleOCR(OCRInstance):
    """
    Paddle-OCR instance
    """
    def __init__(self, lang: str = 'en', kw: Optional[dict] = None) -> None:
        """
        Initialization of Paddle OCR instance
        :param lang: lang parameter used in Paddle
        :param kw: dictionary containing kwargs for PaddleOCR constructor
        """
        if find_spec("paddleocr") is None:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[paddle]' to use this class.")

        # Check paddle version
        paddle_version = version("paddleocr")
        if int(paddle_version.split(".")[0]) >= 3:
            self.instance = PaddleOCR3(lang=lang, kw=kw)
        else:
            self.instance = PaddleOCR2(lang=lang, kw=kw)

    def content(self, document: Document) -> Any:
        return self.instance.content(document=document)

    def to_ocr_dataframe(self, content: Any) -> OCRDataframe:
        return self.instance.to_ocr_dataframe(content=content)
