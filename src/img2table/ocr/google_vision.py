import base64
import binascii
import os
import typing
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe

if typing.TYPE_CHECKING:
    from google.cloud import vision_v1


class VisionContent:
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    @staticmethod
    def img_to_b64(img: np.ndarray) -> str:
        """
        Convert image to base64 string
        :param img: image array
        :return: image in base64 format
        """
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')


class VisionEndpointContent(VisionContent):
    def __init__(self, api_key: str, timeout: int) -> None:
        """
        Document content class from Google Vision using direct requests to endpoint
        :param api_key: Google Vision API key
        :param timeout: requests timeout in seconds
        """
        super().__init__(timeout=timeout)
        self.api_key = api_key

    @staticmethod
    def map_response(response: dict, page: int, width: int, height: int) -> list[dict]:
        """
        Extract test_data from API endpoint response
        :param response: json response from Google API endpoint
        :param page: page number
        :param width: image width
        :param height: image height
        :return: list of OCR elements
        """
        elements = []
        for id_block, block in enumerate(response['responses'][0]['fullTextAnnotation']['pages'][0]['blocks']):
            for id_par, par in enumerate(block.get('paragraphs')):
                id_line = 0
                for id_word, word in enumerate(par.get('words')):
                    # Compute x and y replacement values
                    x_avg = np.mean([el.get('x') for el in word.get('boundingBox').get('vertices') if el.get('x')])
                    x_repl = sorted([0, width], key=lambda val: abs(val - x_avg)).pop(0)
                    y_avg = np.mean([el.get('y') for el in word.get('boundingBox').get('vertices') if el.get('y')])
                    y_repl = sorted([0, height], key=lambda val: abs(val - y_avg)).pop(0)

                    d_el = {
                        "page": page,
                        "class": "ocrx_word",
                        "id": f"word_{id_block}_{id_par}_{id_line}_{id_word}",
                        "parent": f"line_{id_block}_{id_par}_{id_line}",
                        "value": ''.join([sym.get('text') for sym in word.get('symbols')]),
                        "confidence": round(100 * word.get('confidence')),
                        "x1": min(map(lambda el: el.get('x', x_repl), word.get('boundingBox').get('vertices'))),
                        "x2": max(map(lambda el: el.get('x', x_repl), word.get('boundingBox').get('vertices'))),
                        "y1": min(map(lambda el: el.get('y', y_repl), word.get('boundingBox').get('vertices'))),
                        "y2": max(map(lambda el: el.get('y', y_repl), word.get('boundingBox').get('vertices')))
                    }

                    # Check for break
                    _break = word.get('symbols')[-1].get('property', {}).get('detectedBreak', {}).get('type')

                    # Apply breaks
                    if _break in ['EOL_SURE_SPACE', 'LINE_BREAK']:
                        id_line += 1
                    elif _break == 'HYPHEN':
                        id_line += 1
                        d_el['value'] += '-'

                    # Add word to elements
                    elements.append(d_el)

        return elements

    def get_ocr_image(self, img: np.ndarray, page: int) -> list[dict]:
        """
        Extract OCR from image
        :param img: image array
        :param page: page number
        :return: list of OCR elements
        """
        import requests

        # Create payload
        payload = {
            "requests": [
                {
                    "image": {"content": self.img_to_b64(img=img)},
                    "features": [
                        {"type": "DOCUMENT_TEXT_DETECTION"}
                    ]
                }
            ]
        }

        # Post to API
        req = requests.post(url="https://vision.googleapis.com/v1/images:annotate",
                            json=payload,
                            params={"key": self.api_key},
                            timeout=self.timeout)
        response = req.json()

        return self.map_response(response=response, page=page, width=img.shape[1], height=img.shape[0])

    def get_content(self, document: Document) -> list[list[dict]]:
        """
        Get OCR content corresponding to document
        :param document: Document object
        :return: list of OCR elements by page
        """
        # Call API for all images of document
        results = []
        with ThreadPoolExecutor(max_workers=20) as pool:
            args = ((image, idx) for idx, image in enumerate(document.images))
            for ocr in pool.map(lambda d: self.get_ocr_image(*d), args):
                results.append(ocr)

        return results


class VisionAPIContent(VisionContent):
    def __init__(self, timeout: int) -> None:
        """
        Document content class from Google Vision using direct requests to endpoint
        :param timeout: requests timeout in seconds
        """
        try:
            from google.cloud import vision
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[gcp]' to use this class.") from err

        super().__init__(timeout=timeout)
        self.client = vision.ImageAnnotatorClient()

    @staticmethod
    def map_response(response: "vision_v1.types.BatchAnnotateImagesResponse",
                     shapes: list[tuple[int, int]]) -> list[list[dict]]:
        """
        Extract data from API endpoint response object
        :param response: API endpoint response object
        :param shapes: list of images shapes
        :return: list of OCR elements by pages
        """
        from google.cloud import vision_v1

        elements = []
        for id_page, page in enumerate(response.responses):
            # Get image shape
            height, width = shapes[id_page]

            page_elements = []
            for id_block, block in enumerate(page.full_text_annotation.pages[0].blocks):
                for id_par, par in enumerate(block.paragraphs):
                    id_line = 0
                    for id_word, word in enumerate(par.words):
                        # Compute x and y replacement values
                        x_avg = np.mean([el.x for el in word.bounding_box.vertices if hasattr(el, 'x')])
                        x_repl = sorted([0, width], key=lambda val: abs(val - x_avg)).pop(0)
                        y_avg = np.mean([el.y for el in word.bounding_box.vertices if hasattr(el, 'y')])
                        y_repl = sorted([0, height], key=lambda val: abs(val - y_avg)).pop(0)

                        # Compute x and y values in bounding box
                        x_vals = []
                        for vertex in word.bounding_box.vertices:
                            try:
                                x_vals.append(vertex.x or x_repl)
                            except AttributeError:
                                x_vals.append(x_repl)

                        y_vals = []
                        for vertex in word.bounding_box.vertices:
                            try:
                                y_vals.append(vertex.y or y_repl)
                            except AttributeError:
                                y_vals.append(y_repl)

                        d_el = {
                            "page": id_page,
                            "class": "ocrx_word",
                            "id": f"word_{id_block}_{id_par}_{id_line}_{id_word}",
                            "parent": f"line_{id_block}_{id_par}_{id_line}",
                            "value": ''.join([sym.text for sym in word.symbols]),
                            "confidence": round(100 * word.confidence),
                            "x1": min(x_vals),
                            "x2": max(x_vals),
                            "y1": min(y_vals),
                            "y2": max(y_vals)
                        }

                        # Check for break
                        _break = word.symbols[-1].property.detected_break.type

                        # Apply breaks
                        if _break in [vision_v1.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                                      vision_v1.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK]:
                            id_line += 1
                        elif _break == vision_v1.TextAnnotation.DetectedBreak.BreakType.HYPHEN:
                            id_line += 1
                            d_el['value'] += '-'

                        page_elements.append(d_el)

            elements.append(page_elements)

        return elements

    def get_content(self, document: Document) -> list[list[dict]]:
        """
        Get OCR content corresponding to document
        :param document: Document object
        :return: list of OCR elements by page
        """
        try:
            from google.cloud import vision_v1
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError("Missing dependencies, please install 'img2table[gcp]' to use this class.") from err

        reqs, shapes = [], []
        for img in document.images:
            # Create image object
            image = vision_v1.Image()
            image.content = binascii.a2b_base64(self.img_to_b64(img=img))

            shapes.append(img.shape[:2])

            # Create request
            request = vision_v1.AnnotateImageRequest()
            request.image = image
            request.features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]

            reqs.append(request)

        # Call API
        result = self.client.batch_annotate_images(requests=reqs,
                                                   timeout=self.timeout)

        return self.map_response(response=result, shapes=shapes)


class VisionOCR(OCRInstance):
    """
    Google Vision OCR instance
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15) -> None:
        """
        Initialization of Google Vision OCR instance
        :param api_key: Google Vision API key
        :param timeout: requests timeout in seconds
        """
        # Extract GCP credentials
        gcp_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # Validation on input and environment
        if not (isinstance(api_key, str) or api_key is None):
            raise TypeError(f"Invalid type {type(api_key)} for api_key argument")

        # If no API key is provided, check for "GOOGLE_APPLICATION_CREDENTIALS" env variable
        if gcp_credentials is None and api_key is None:
            raise ValueError('The GOOGLE_APPLICATION_CREDENTIALS environment variable should be set if no API key '
                             'is provided')

        # Instantiate content_getter
        if gcp_credentials:
            self.content_getter = VisionAPIContent(timeout=timeout)
        else:
            self.content_getter = VisionEndpointContent(api_key=api_key, timeout=timeout)

    def content(self, document: Document) -> list[list[dict]]:
        return self.content_getter.get_content(document=document)

    def to_ocr_dataframe(self, content: list[list[dict]]) -> OCRDataframe:
        """
        Convert list of OCR elements by page to OCRDataframe object
        :param content: list of OCR elements by page
        :return: OCRDataframe object corresponding to content
        """
        list_dfs = [pl.DataFrame(data=page_elements, schema=self.pl_schema)
                    for page_elements in content if page_elements]

        return OCRDataframe(df=pl.concat(list_dfs)) if list_dfs else None
