# coding: utf-8

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from tesserocr import PyTessBaseAPI, PSM

from img2table.document import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe


class TesseractOCR(OCRInstance):
    def __init__(self, *args, n_threads: int = os.cpu_count(), lang: str = 'eng', **kwargs):
        """
        Initialization of Tesseract OCR instance
        :param n_threads: number of parallel threads used for Tesseract
        :param lang: lang parameter used in Tesseract
        """
        super().__init__(*args, **kwargs)
        self.lang = lang
        self.n_threads = n_threads

    def hocr(self, image: np.ndarray, page_number: int = 0) -> str:
        """
        Get hOCR HTML of an image using Tesseract
        :param image: numpy array representing the image
        :param page_number: page index
        :return: hOCR HTML string
        """
        with PyTessBaseAPI(lang=self.lang, psm=PSM.SPARSE_TEXT) as api:
            # Convert image to PIL
            pil_img = Image.fromarray(obj=image)

            # Get hocr
            api.SetImage(pil_img)
            hocr = api.GetHOCRText(page_number)

        return hocr

    def content(self, document: Document) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            args = [(img, idx) for idx, img in enumerate(document.images)]
            hocrs = pool.map(lambda d: self.hocr(*d), args)

        return hocrs

    def to_ocr_dataframe(self, content: List[str]) -> OCRDataframe:
        """
        Convert hOCR HTML to OCRDataframe object
        :param content: hOCR HTML string
        :return: OCRDataframe object corresponding to content
        """
        # Create list of dataframes for each page
        list_dfs = list()

        for hocr in content:
            # Instantiate HTML parser
            soup = BeautifulSoup(hocr, features='html.parser')

            # Parse all HTML elements
            list_elements = list()
            for element in soup.find_all(class_=True):
                # Get element properties
                d_el = {
                    "class": element["class"][0],
                    "id": element["id"],
                    "parent": element.parent.get('id'),
                    "value": re.sub(r"^(\s|\||L|_|;|\*)*$", '', element.string).strip() or None if element.string else None
                }

                # Get word confidence
                str_conf = re.findall(r"x_wconf \d{1,2}", element["title"])
                if str_conf:
                    d_el["confidence"] = int(str_conf[0].split()[1])
                else:
                    d_el["confidence"] = np.nan

                # Get bbox
                bbox = re.findall(r"bbox \d{1,4} \d{1,4} \d{1,4} \d{1,4}", element["title"])[0]
                d_el["x1"], d_el["y1"], d_el["x2"], d_el["y2"] = tuple(
                    int(element) for element in re.sub(r"^bbox\s", "", bbox).split())

                list_elements.append(d_el)

            # Create dataframe
            list_dfs.append(pd.DataFrame(list_elements))

        return OCRDataframe(df=pd.concat(list_dfs))
