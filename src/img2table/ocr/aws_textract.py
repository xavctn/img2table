from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from img2table.ocr._types import OCRData, OCRInstance

if TYPE_CHECKING:
    import numpy as np

    from img2table.document._types import Document, MockDocument


class TextractOCR(OCRInstance):
    """
    AWS Textract instance
    """

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region: str | None = None,
    ) -> None:
        """
        Initialization of AWS Textract OCR instance
        :param aws_access_key_id: AWS access key id
        :param aws_secret_access_key: AWS secret access key
        :param aws_session_token: AWS temporary session token
        :param region: AWS server region
        """
        try:
            import boto3
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Missing dependencies, please install 'img2table[aws]' to use this class."
            ) from err

        if not any(
            v is None for v in [aws_access_key_id, aws_secret_access_key, aws_session_token]
        ):
            self.client = boto3.client(
                service_name="textract",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region,
            )
        else:
            self.client = boto3.client(service_name="textract", region_name=region)

    @staticmethod
    def map_response(response: dict, image: np.ndarray) -> list[dict]:
        """
        Extract data from API endpoint response dictionary
        :param response: dictionary returned by Textract API
        :param image: image array
        :return: list of OCR elements corresponding to the page
        """
        # Get image dimensions
        height, width = image.shape[:2]

        # Initialize dictionary containing child relationships
        dict_children = {}

        # Parse blocks and identify words
        word_elements = []
        for block in response["Blocks"]:
            # Identify children and add relationship to dict
            children = [
                child
                for rel in block.get("Relationships", [])
                for child in rel.get("Ids")
                if rel.get("Type") == "CHILD"
            ]
            for child in children:
                dict_children[child] = block.get("Id")

            # If the block is a word, parse characteristics and add to word_elements
            if block.get("BlockType") == "WORD":
                d_block = {
                    "id": block.get("Id"),
                    "parent": dict_children.get(block.get("Id")),
                    "value": block.get("Text"),
                    "confidence": round(block.get("Confidence", 0)),
                    "x1": round(
                        min(el.get("X") for el in block.get("Geometry").get("Polygon")) * width
                    ),
                    "x2": round(
                        max(el.get("X") for el in block.get("Geometry").get("Polygon")) * width
                    ),
                    "y1": round(
                        min(el.get("Y") for el in block.get("Geometry").get("Polygon")) * height
                    ),
                    "y2": round(
                        max(el.get("Y") for el in block.get("Geometry").get("Polygon")) * height
                    ),
                }
                word_elements.append(d_block)

        return word_elements

    def of(self, document: Document | MockDocument) -> OCRData | None:
        """
        Get OCR content corresponding to document
        :param document: Document object
        :return: list of OCR elements by page
        """
        records = {}
        for page, image in enumerate(document.images):
            _, img = cv2.imencode(".jpg", image)
            content = self.client.detect_document_text(Document={"Bytes": img.tobytes()})
            records[page] = self.map_response(response=content, image=image)

        return OCRData(records=records) if records else None
