
import typing

import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe

if typing.TYPE_CHECKING:
    import doctr


class DocTR(OCRInstance):
    """
    DocTR instance
    """
    def __init__(self, detect_language: bool = False, kw: typing.Optional[dict] = None) -> None:
        """
        Initialization of EasyOCR instance
        """
        try:
            from doctr.models import ocr_predictor
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError("Missing dependencies, please install doctr to use this class.") from err

        # Create kwargs dict for constructor
        kw = kw or {}
        kw["detect_language"] = detect_language
        kw["pretrained"] = kw.get("pretrained") if kw.get("pretrained") is not None else True

        self.model = ocr_predictor(**kw)

    def content(self, document: Document) -> "doctr.io.elements.Document":
        # Get OCR of all images
        return self.model(document.images)

    def to_ocr_dataframe(self, content: "doctr.io.elements.Document") -> OCRDataframe:
        """
        Convert docTR Document object to OCRDataframe object
        :param content: docTR Document object
        :return: OCRDataframe object corresponding to content
        """
        # Create list of elements
        list_elements = []

        for page_id, page in enumerate(content.pages):
            dimensions = page.dimensions
            word_id = 0
            for block in page.blocks:
                for line_id, line in enumerate(block.lines):
                    for word in line.words:
                        word_id += 1
                        dict_word = {
                            "page": page_id,
                            "class": "ocrx_word",
                            "id": f"word_{page_id + 1}_{line_id}_{word_id}",
                            "parent": f"word_{page_id + 1}_{line_id}",
                            "value": word.value,
                            "confidence": round(100 * word.confidence),
                            "x1": round(word.geometry[0][0] * dimensions[1]),
                            "y1": round(word.geometry[0][1] * dimensions[0]),
                            "x2": round(word.geometry[1][0] * dimensions[1]),
                            "y2": round(word.geometry[1][1] * dimensions[0])
                        }

                        list_elements.append(dict_word)

        return OCRDataframe(df=pl.DataFrame(list_elements)) if list_elements else None
