# coding: utf-8

import os
import warnings
from tempfile import NamedTemporaryFile
from typing import List, Dict

import cv2
import numpy as np
import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe


class PaddleOCR(OCRInstance):
    """
    Paddle-OCR instance
    """
    def __init__(self, lang: str = 'en', kw: Dict = None):
        """
        Initialization of Paddle OCR instance
        :param lang: lang parameter used in Paddle
        :param kw: dictionary containing kwargs for PaddleOCR constructor
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from paddleocr import PaddleOCR as OCR
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[paddle]' to use this class.")

        if isinstance(lang, str):
            self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        # Create kwargs dict for constructor
        kw = kw or {}
        kw["lang"] = self.lang
        kw["use_angle_cls"] = kw.get("use_angle_cls") or False
        kw["show_log"] = kw.get("show_log") or False

        self.ocr = OCR(**kw)

    def hocr(self, image: np.ndarray) -> List:
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
        while os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except PermissionError:
                pass

        # Get result
        ocr_result = ocr_result.pop()
        return [[bbox, (word[0], round(word[1], 2))] for bbox, word in ocr_result] if ocr_result else []

    def content(self, document: Document) -> List[List]:
        # Get OCR of all images
        ocrs = [self.hocr(image=image) for image in document.images]

        return ocrs

    def to_ocr_dataframe(self, content: List[List]) -> OCRDataframe:
        """
        Convert hOCR HTML to OCRDataframe object
        :param content: hOCR HTML string
        :return: OCRDataframe object corresponding to content
        """
        # Create list of elements
        list_elements = list()

        for page, ocr_result in enumerate(content):
            word_id = 0
            for bbox, word in ocr_result:
                word_id += 1
                dict_word = {
                    "page": page,
                    "class": "ocrx_word",
                    "id": f"word_{page + 1}_{word_id}",
                    "parent": f"word_{page + 1}_{word_id}",
                    "value": word[0],
                    "confidence": 100 * word[1],
                    "x1": round(min([edge[0] for edge in bbox])),
                    "y1": round(min([edge[1] for edge in bbox])),
                    "x2": round(max([edge[0] for edge in bbox])),
                    "y2": round(max([edge[1] for edge in bbox]))
                }

                list_elements.append(dict_word)

        return OCRDataframe(df=pl.LazyFrame(list_elements, schema=self.pl_schema)) if list_elements else None
