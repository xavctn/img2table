# coding: utf-8
import os
import time
import typing
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe

if typing.TYPE_CHECKING:
    from azure.cognitiveservices.vision.computervision.models import ReadOperationResult

class AzureDocumentIntelligence(OCRInstance):
    """
    Azure Document Intelligence READ instance
    This uses the Document Intelligence 'prebuilt-read' model
    """
    def __init__(self, endpoint: Optional[str] = None, subscription_key: Optional[str] = None):
        """
        Initialization of Azure Document Intelligence OCR instance
        :param endpoint: Azure Document Intelligence inference endpoint
        :param subscription_key: Azure Document Intelligence subscription key
        """
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
            from azure.identity import DefaultAzureCredential
            from azure.ai.documentintelligence.models import AnalyzeResult
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[azure]' to use this class.")

        # Validation on endpoint variable
        if not (isinstance(endpoint, str) or endpoint is None):
            raise TypeError(f"Invalid type {type(endpoint)} for endpoint argument")

        endpoint = endpoint or os.getenv('DOCUMENT_INTELLIGENCE_ENDPOINT')
        if endpoint is None:
            raise ValueError('The DOCUMENT_INTELLIGENCE_ENDPOINT environment variable should be set if no endpoint '
                             'is provided')

        # Validation on subscription_key variable
        if not (isinstance(subscription_key, str) or subscription_key is None):
            raise TypeError(f"Invalid type {type(subscription_key)} for endpoint argument")

        subscription_key = subscription_key or os.getenv('DOCUMENT_INTELLIGENCE_SUBSCRIPTION_KEY')
        
        # Create client -- use AzureKeyCredential if subscription key is provided, otherwise use DefaultAzureCredential
        if subscription_key:
            self.client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(subscription_key)
            )
        else:
            self.client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=DefaultAzureCredential()
            )

    def content(self, document: Document) -> Tuple[List[dict], List[dict]]:
        """
        Extract document text using Azure Document Intelligence API
        :param document: Document object
        :return: list of Azure Document Intelligence API results
        """

        page_words = []
        page_paragraphs = []

        pollers = []
        for image in document.images:
            # Encode image as JPEG
            _, img_bytes = cv2.imencode(".jpg", image)
            img_stream = BytesIO(img_bytes.tobytes())
            poller = self.client.begin_analyze_document(
                "prebuilt-read",
                body=img_stream
            )
            pollers.append(poller)

        # Poll until all results are ready
        results = [poller.result() for poller in pollers]
        while not all(hasattr(r, 'pages') and r['pages'] for r in results):
            time.sleep(0.1)
            results = [poller.result() for poller in pollers]

        for result in results:
            page = result['pages'][0]
            page_words.append(page['words'])
            page_paragraphs.append(result['paragraphs'])

        return page_words, page_paragraphs
    
    def to_ocr_dataframe(self, content: Tuple[List[dict], List[dict]]) -> OCRDataframe:
        """
        Convert list of OCR results by page to OCRDataframe object
        :param content: tuple containing list of words, page widths, page heights, and page angles
        :return: OCRDataframe object corresponding to content
        """
        words, paragraphs = content
        list_dfs = []

        for page_idx, (page_words, page_paragraphs) in enumerate(zip(words, paragraphs)):
            word_elements = []
            word_cnt = 0

            for word in page_words:
                word_cnt += 1
                bbox = word['polygon']
                d_word = {
                    "page": page_idx,
                    "class": "ocrx_word",
                    "id": f"word_{page_idx + 1}_{word_cnt}",
                    "parent": f"word_{page_idx + 1}_{self._find_parent_id(word, page_paragraphs) + 1}",
                    "value": word['content'],
                    "confidence": round(100 * word['confidence']),
                    "x1": min(bbox[::2]),
                    "x2": max(bbox[::2]),
                    "y1": min(bbox[1::2]),
                    "y2": max(bbox[1::2])
                }
                word_elements.append(d_word)

            if word_elements:
                list_dfs.append(pl.DataFrame(data=word_elements, schema=self.pl_schema))

        return OCRDataframe(df=pl.concat(list_dfs)) if list_dfs else None
    
    def _find_parent_id(self, word: dict, paragraphs_list: List[dict]) -> Optional[int]:
        """
        Find the parent paragraph for a given word.
        Criteria:
          - The word's centroid is within the paragraph's bounding box.
          - The word's content is a substring of the paragraph's content.
        Returns the paragraph index if found, else raises ValueError.
        """
        def centroid(bbox):
            xs = bbox[::2]
            ys = bbox[1::2]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
    
        def centroid_within(cx, cy, para_bbox):
            px1, px2 = min(para_bbox[::2]), max(para_bbox[::2])
            py1, py2 = min(para_bbox[1::2]), max(para_bbox[1::2])
            return (px1 <= cx <= px2) and (py1 <= cy <= py2)
    
        cx, cy = centroid(word['polygon'])
    
        for para_idx, para in enumerate(paragraphs_list):
            para_content = para.get('content', '')
            for region in para.get('boundingRegions', []):
                para_bbox = region.get('polygon', [])
                if para_bbox and centroid_within(cx, cy, para_bbox):
                    if word['content'] in para_content:
                        return para_idx
    
        raise ValueError(f"Could not find parent paragraph for word: {word['content']} with centroid ({cx}, {cy})")

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
        try:
            from azure.cognitiveservices.vision.computervision import ComputerVisionClient
            from msrest.authentication import CognitiveServicesCredentials
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[azure]' to use this class.")

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

    def content(self, document: Document) -> List["ReadOperationResult"]:
        """
        Extract document text using Azure OCR API
        :param document: Document object
        :return: list of Azure OCR API results
        """
        try:
            from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[azure]' to use this class.")

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

    def to_ocr_dataframe(self, content: List["ReadOperationResult"]) -> OCRDataframe:
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

            if word_elements:
                list_dfs.append(pl.DataFrame(data=word_elements, schema=self.pl_schema))

        return OCRDataframe(df=pl.concat(list_dfs)) if list_dfs else None
