# coding: utf-8
import os
import time
from io import BytesIO
from typing import List, Optional

import cv2
import polars as pl
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import ReadOperationResult, OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe


class AzureOCR(OCRInstance):
    """
    Azure Cognitive Services OCR instance
    """

    def __init__(self, endpoint: Optional[str] = None, subscription_key: Optional[str] = None):
        """
        Initialization of Azure Cognitive Services OCR instance
        :param endpoint: Azure Cognitive Services endpoint
        :param subscription_key: Azure Cognitive Services subscription key
        """
        # Validation on endpoint variable
        if not (isinstance(endpoint, str) or endpoint is None):
            raise TypeError(f"Invalid type {type(endpoint)} for endpoint argument")

        endpoint = endpoint or os.getenv('COMPUTER_VISION_ENDPOINT')
        if endpoint is None:
            raise ValueError('The COMPUTER_VISION_ENDPOINT environment variable should be set if no endpoint '
                             'is provided')

        # Validation on subscription_key variable
        if not (isinstance(subscription_key, str) or subscription_key is None):
            raise TypeError(f"Invalid type {type(subscription_key)} for endpoint argument")

        subscription_key = subscription_key or os.getenv('COMPUTER_VISION_SUBSCRIPTION_KEY')
        if subscription_key is None:
            raise ValueError('The COMPUTER_VISION_SUBSCRIPTION_KEY environment variable should be set if no API key '
                             'is provided')

        self.client = ComputerVisionClient(endpoint=endpoint,
                                           credentials=CognitiveServicesCredentials(subscription_key=subscription_key))

    def content(self, document: Document) -> List[ReadOperationResult]:
        """
        Extract document text using Azure OCR API
        :param document: Document object
        :return: list of Azure OCR API results
        """
        # Create list of file-like images
        images = list()
        for image in document.images:
            _, img = cv2.imencode(".jpg", image)
            images.append(BytesIO(img.tobytes()))

        # Call API and get operation IDs
        operations_ids = [self.client.read_in_stream(image=image, raw=True).headers.get('Operation-Location').split('/')[-1]
                          for image in images]

        # Retrieve results
        results = [self.client.get_read_result(operation_id) for operation_id in operations_ids]
        while not all(map(lambda r: r.status == OperationStatusCodes.succeeded, results)):
            time.sleep(0.1)
            results = [self.client.get_read_result(operation_id) for operation_id in operations_ids]

        return results

    def to_ocr_dataframe(self, content: List[ReadOperationResult]) -> OCRDataframe:
        """
        Convert list of OCR results by page to OCRDataframe object
        :param content: list of OCR results by page
        :return: OCRDataframe object corresponding to content
        """
        list_dfs = list()

        # Parse all page results
        for page, result in enumerate(content):
            word_elements = list()
            line_cnt = 0
            word_cnt = 0
            for r in result.analyze_result.read_results:
                for line in r.lines:
                    line_cnt += 1
                    for word in line.words:
                        word_cnt += 1

                        bbox = list(map(int, word.bounding_box))
                        d_word = {
                            "page": page,
                            "class": "ocrx_word",
                            "id": f"word_{page + 1}_{word_cnt}",
                            "parent": f"word_{page + 1}_{line_cnt}",
                            "value": word.text,
                            "confidence": round(100 * word.confidence),
                            "x1": min(bbox[::2]),
                            "x2": max(bbox[::2]),
                            "y1": min(bbox[1::2]),
                            "y2": max(bbox[1::2])
                        }

                        word_elements.append(d_word)

            list_dfs.append(pl.from_dicts(word_elements))

        return OCRDataframe(df=pl.concat(list_dfs).lazy())
