# coding: utf-8

import os
import warnings
from tempfile import NamedTemporaryFile
from typing import List

import cv2
import numpy as np
import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from paddleocr import PaddleOCR as OCR


class PaddleOCR(OCRInstance):
    """
    Paddle-OCR instance
    """
    def __init__(self, lang: str = 'en'):
        """
        Initialization of Paddle OCR instance
        :param lang: lang parameter used in Paddle
        """
        if isinstance(lang, str):
            self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        self.ocr = OCR(lang=self.lang, use_angle_cls=False, show_log=False)

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

        return [[bbox, (word[0], round(word[1], 2))] for bbox, word in ocr_result.pop()]

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
                    "page": 0,
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

        return OCRDataframe(df=pl.LazyFrame(list_elements)) if list_elements else None
