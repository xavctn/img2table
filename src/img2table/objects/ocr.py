# coding: utf-8
import re
from typing import List, Tuple

from bs4 import BeautifulSoup


class OCRObject(object):
    html_tag: str
    html_classes: List
    item_class = None

    def __init__(self, html_title: str, items: List = None):
        self._items = items or []
        self._title = html_title
        self._bbox = self.bbox_from_title()

    @property
    def items(self) -> List:
        return self._items

    @property
    def bbox(self):
        return self._bbox

    def bbox_from_title(self) -> tuple:
        """
        Extract bounding box from HOCR html title
        :return: bounding box
        """
        # Extract bbox part
        bbox_string = re.findall(r"bbox \d{1,4} \d{1,4} \d{1,4} \d{1,4}", self._title)[0]

        return tuple(int(element) for element in re.sub(r"^bbox\s", "", bbox_string).split())

    def add_items(self, item):
        """
        Add word to line
        :param item: OCRObject object or list
        :return:
        """
        if isinstance(item, list):
            self._items += item
        else:
            self._items += [item]

    def get_items(self, soup: BeautifulSoup):
        list_items = list()
        for html_element in soup.find_all(self.item_class.html_tag, {"class": self.item_class.html_classes}):
            item = self.item_class(html_title=html_element.get('title'))
            item.get_items(soup=html_element)
            list_items.append(item)
        self.add_items(item=list_items)

    def intersect_bbox(self, bbox: tuple) -> bool:
        """
        Assess if a bounding box intersects the object
        :param bbox: tuple representing a bounding box
        :return: boolean indicating if the bounding box intersects the object
        """
        # determine the coordinates of the intersection rectangle
        x_left = max(self.bbox[0], bbox[0])
        y_top = max(self.bbox[1], bbox[1])
        x_right = min(self.bbox[2], bbox[2])
        y_bottom = min(self.bbox[3], bbox[3])

        if x_right < x_left or y_bottom < y_top:
            return False

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        bb2_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou > 0

    def is_contained_in_bbox(self, bbox: tuple):
        """
        Assess if the object is contained in a bounding box
        :param bbox: tuple representing a bounding box
        :return: boolean indicating if the object is contained in the bounding box
        """
        # determine the coordinates of the intersection rectangle
        x_left = max(self.bbox[0], bbox[0])
        y_top = max(self.bbox[1], bbox[1])
        x_right = min(self.bbox[2], bbox[2])
        y_bottom = min(self.bbox[3], bbox[3])

        if x_right < x_left or y_bottom < y_top:
            return False

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

        return intersection_area / bb1_area >= 0.75

    def get_bbox_items(self, bbox: tuple):
        # Get items that are intersecting with bounding box
        intersecting_items = sort_ocr_objects([item for item in self.items if item.intersect_bbox(bbox=bbox)])
        return [sub_item for item in intersecting_items for sub_item in item.get_bbox_items(bbox=bbox) if sub_item]


def sort_ocr_objects(ocr_objects: List[OCRObject], vertically: bool = True) -> List[OCRObject]:
    """
    Sort OCR objects list vertically and horizontally
    :param ocr_objects: list of OCR objects
    :param vertically: boolean to sort vertically, if false, list will be sorted horizontally
    :return: sorted OCR objects list
    """
    if vertically:
        sorted_objects = sorted(ocr_objects,
                                key=lambda el: (el.bbox[1], el.bbox[3], el.bbox[0], el.bbox[2]))
    else:
        sorted_objects = sorted(ocr_objects,
                                key=lambda el: (el.bbox[0], el.bbox[2], el.bbox[1], el.bbox[3]))

    return sorted_objects


class OCRWord(OCRObject):
    html_tag = "span"
    html_classes = ["ocrx_word"]
    item_class = None

    def __init__(self, html_title: str, value: str = None, items: List = None):
        super().__init__(html_title=html_title, items=items)
        self._value = value

    @property
    def value(self):
        if re.search(r"^(—|L|=|\!|_|\||\[|\]|I|l|\.)*$", self._value) is None:
            return self._value
        return None

    def get_items(self, soup: BeautifulSoup):
        self._value = soup.string


class OCRLine(OCRObject):
    html_tag = "span"
    html_classes = ["ocr_line", "ocr_caption", "ocr_header", "ocr_textfloat"]
    item_class = OCRWord

    def __init__(self, html_title: str, items: List = None):
        super().__init__(html_title=html_title, items=items)
        self._x_size = self.x_size_from_title()

    @property
    def x_size(self):
        return self._x_size

    def x_size_from_title(self) -> int:
        """
        Extract x_size from HOCR html title
        :return: x_size
        """
        return round(float(re.findall(r"(x_size )([\d\.]+)", self._title)[0][1]))

    def get_bbox_items(self, bbox: tuple) -> List[Tuple[List[OCRObject], int]]:
        """
        Get list of words that are contained in the bounding box, as well as line height
        :param bbox: bounding box
        :return:
        """
        # Get items that are intersecting with bounding box
        intersecting_words = sort_ocr_objects([word for word in self.items
                                               if word.is_contained_in_bbox(bbox=bbox)
                                               and word.value is not None],
                                              vertically=False)
        # Keep only non empty lines
        intersecting_words = [word_line for word_line in intersecting_words if word_line]

        return [(intersecting_words, self._x_size)]


class OCRParagraph(OCRObject):
    html_tag = "p"
    html_classes = ["ocr_par"]
    item_class = OCRLine


class OCRArea(OCRObject):
    html_tag = "div"
    html_classes = ["ocr_carea"]
    item_class = OCRParagraph


class OCRPage(OCRObject):
    html_tag = "div"
    html_classes = ["ocr_page"]
    item_class = OCRArea

    @classmethod
    def parse_hocr(cls, hocr_html: str):
        """
        Parse HOCR html to OCRPage object
        :param hocr_html: HOCR html
        :return: OCRPage object
        """
        # Instantiate HTML parser
        soup = BeautifulSoup(hocr_html, features='html')

        # Instantiate OCR page
        for html_page in soup.find_all("div", {"class": "ocr_page"}):
            html_title = html_page.get('title')
        ocr_page = cls(html_title=html_title)
        ocr_page.get_items(soup=html_page)

        return ocr_page

    def get_text_cell(self, cell, margin: int = 0) -> str:
        """
        Get text corresponding to cell
        :param cell: Cell object in document
        :param margin: margin to take around cell
        :return: text contained in cell
        """
        # Define relevant bounding box
        bbox = cell.bbox(margin=margin)

        # Get word lines corresponding to bounding box
        cell_word_lines = self.get_bbox_items(bbox=bbox)
        word_values = [[word.value for word in line] for line, x_size in cell_word_lines]

        # Create strings at line level and general string
        line_strings = [" ".join(line) for line in word_values]
        final_string = "\n".join(line_strings)

        return final_string.strip() or None

    def get_text_sizes(self, cell, margin: int = 0) -> List[int]:
        """
        Get text sized corresponding to cell
        :param cell: cell in document
        :param margin: margin to take around cell
        :return: list of text sizes contained in cell
        """
        # Define relevant bounding box
        bbox = cell.bbox(margin=margin)

        # Get word lines corresponding to bounding box
        cell_word_lines = self.get_bbox_items(bbox=bbox)

        # Create list of all word sizes
        word_sizes = [x_size for line, x_size in cell_word_lines for word in line]

        return word_sizes
