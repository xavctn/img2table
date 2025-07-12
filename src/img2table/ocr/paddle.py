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
from packaging import version

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
                from paddleocr import PaddleOCR as OCR , __version__ 
                self.version = __version__
                
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[paddle]' to use this class.")

        if isinstance(lang, str):
            self.lang = lang
        else:
            raise TypeError(f"Invalid type {type(lang)} for lang argument")

        def init_2x(self,lang: str = 'en', kw: Dict = None):
           
            # Create kwargs dict for constructor
            kw = kw or {}
            kw["lang"] = lang
            kw["use_angle_cls"] = kw.get("use_angle_cls") or False
            kw["show_log"] = kw.get("show_log") or False

            self.ocr = OCR(**kw)

        def init_3x(self,lang: str = 'en', kw: Dict = None):
            kw = kw or {}
            kw["lang"] = lang
            kw["use_angle_cls"] = kw.get("use_angle_cls") or False
            
            self.ocr = OCR(**kw)
            
        if version.parse(self.version) >= version.parse("3.0.0"):
            init_3x(self=self,lang=lang,kw=kw)
        else:
            init_2x(self=self,lang=lang,kw=kw)
    
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

            # Get OCR and the version
            if version.parse(self.version) >= version.parse("3.0.0"):
                ocr_result = self.ocr.predict(tmp_file)
            else:
                ocr_result = self.ocr.ocr(img=tmp_file, cls=False)

        # Remove temporary file
        while os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except PermissionError:
                pass
        # Get result
        ocr_result = ocr_result.pop()
        if version.parse(self.version) >= version.parse("3.0.0"):
            results = self.hocr_3x(ocr_result)
        else:
            results = [[bbox, (word[0], round(word[1], 2))] for bbox, word in ocr_result] if ocr_result else []
        return  results

    def hocr_3x(self,ocr_result):
        """
        PaddleOCR > 3.0.0 to hadnle the results
        """
        def box_to_points(box):
            x1, y1, x2, y2 = box
            return np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])
        lines = list(zip(ocr_result["rec_boxes"], zip(ocr_result["rec_texts"], ocr_result["rec_scores"]))) 
        normalized_lines = []
        for bbox, (text, score) in lines:
            if isinstance(bbox, np.ndarray) and bbox.shape == (4,):
                bbox = box_to_points(bbox)
                normalized_lines.append((bbox, (text, score)))

        return [[bbox, (text, round(score, 2))] for bbox, (text, score) in normalized_lines] if normalized_lines else []
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

        return OCRDataframe(df=pl.DataFrame(list_elements, schema=self.pl_schema)) if list_elements else None
