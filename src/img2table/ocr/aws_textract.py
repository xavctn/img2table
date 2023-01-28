# coding: utf-8

from typing import List, Iterator, Optional, Dict

import boto3
import cv2
import numpy as np
import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe


class TextractOCR(OCRInstance):
    """
    AWS Textract instance
    """
    def __init__(self, aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None, region: Optional[str] = None):
        """
        Initialization of AWS Textract OCR instance
        :param aws_access_key_id: AWS access key id
        :param aws_secret_access_key: AWS secret access key
        :param aws_session_token: AWS temporary session token
        :param region: AWS server region
        """
        if not any(map(lambda v: v is None, [aws_access_key_id, aws_secret_access_key, aws_session_token])):
            self.client = boto3.client(service_name='textract',
                                       aws_access_key_id=aws_access_key_id,
                                       aws_secret_access_key=aws_secret_access_key,
                                       aws_session_token=aws_session_token,
                                       region_name=region)
        else:
            self.client = boto3.client(service_name='textract',
                                       region_name=region)

    @staticmethod
    def map_response(response: Dict, image: np.ndarray, page: int) -> List[Dict]:
        """
        Extract data from API endpoint response dictionary
        :param response: dictionary returned by Textract API
        :param image: image array
        :param page: page number
        :return: list of OCR elements corresponding to the page
        """
        # Get image dimensions
        height, width = image.shape

        # Initialize dictionary containing child relationships
        dict_children = dict()

        # Parse blocks and identify words
        word_elements = list()
        for block in response.get('Blocks'):
            # Identify children and add relationship to dict
            children = [child for rel in block.get('Relationships', []) for child in rel.get('Ids')
                        if rel.get('Type') == 'CHILD']
            for child in children:
                dict_children[child] = block.get('Id')

            # If the block is a word, parse characteristics and add to word_elements
            if block.get('BlockType') == "WORD":
                d_block = {
                    "page": page,
                    "class": "ocrx_word",
                    "id": block.get('Id'),
                    "parent": dict_children.get(block.get('Id')),
                    "value": block.get("Text"),
                    "confidence": round(block.get('Confidence', 0)),
                    "x1": round(min(map(lambda el: el.get('X'), block.get('Geometry').get('Polygon'))) * width),
                    "x2": round(max(map(lambda el: el.get('X'), block.get('Geometry').get('Polygon'))) * width),
                    "y1": round(min(map(lambda el: el.get('Y'), block.get('Geometry').get('Polygon'))) * height),
                    "y2": round(max(map(lambda el: el.get('Y'), block.get('Geometry').get('Polygon'))) * height),
                }
                word_elements.append(d_block)

        return word_elements

    def content(self, document: Document) -> Iterator[List[Dict]]:
        """
        Get OCR content corresponding to document
        :param document: Document object
        :return: list of OCR elements by page
        """
        for page, image in enumerate(document.images):
            _, img = cv2.imencode(".jpg", image)
            content = self.client.detect_document_text(Document={'Bytes': img.tobytes()})
            yield self.map_response(response=content, image=image, page=page)

    def to_ocr_dataframe(self, content: Iterator[List[Dict]]) -> OCRDataframe:
        """
        Convert list of OCR elements by page to OCRDataframe object
        :param content: list of OCR elements by page
        :return: OCRDataframe object corresponding to content
        """
        list_dfs = list(map(pl.from_dicts, content))

        return OCRDataframe(df=pl.concat(list_dfs).lazy())
