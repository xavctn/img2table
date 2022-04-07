# coding: utf-8
import re

import numpy as np
import pandas as pd
import pytesseract
from bs4 import BeautifulSoup
from cv2 import cv2


class OCRPage(object):
    def __init__(self, hocr_html: str):
        self._hocr_html = hocr_html
        self._df = self.parse_hocr()

    @property
    def hocr_html(self) -> str:
        return self._hocr_html

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def parse_hocr(self) -> pd.DataFrame:
        # Instantiate HTML parser
        soup = BeautifulSoup(self.hocr_html, features='html.parser')

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
        return pd.DataFrame(list_elements)

    def get_text_cell(self, cell, margin: int = 0) -> str:
        """
        Get text corresponding to cell
        :param cell: Cell object in document
        :param margin: margin to take around cell
        :return: text contained in cell
        """
        # Define relevant bounding box
        bbox = cell.bbox(margin=margin)

        # Filter dataframe on non empty words
        df_words = self.df[self.df["class"] == "ocrx_word"]
        df_words = df_words[df_words["value"].notnull()]

        # Compute coordinates of intersection
        df_words = df_words.assign(**{"x1_bbox": bbox[0],
                                      "y1_bbox": bbox[1],
                                      "x2_bbox": bbox[2],
                                      "y2_bbox": bbox[3]})
        df_words["x_left"] = df_words[["x1", "x1_bbox"]].max(axis=1)
        df_words["y_top"] = df_words[["y1", "y1_bbox"]].max(axis=1)
        df_words["x_right"] = df_words[["x2", "x2_bbox"]].min(axis=1)
        df_words["y_bottom"] = df_words[["y2", "y2_bbox"]].min(axis=1)

        # Filter where intersection is not empty
        df_words = df_words[df_words["x_right"] > df_words["x_left"]]
        df_words = df_words[df_words["y_bottom"] > df_words["y_top"]]

        # Compute area of word bbox and intersection
        df_words["w_area"] = (df_words["x2"] - df_words["x1"]) * (df_words["y2"] - df_words["y1"])
        df_words["int_area"] = (df_words["x_right"] - df_words["x_left"]) * (df_words["y_bottom"] - df_words["y_top"])

        # Filter on words where its bbox is contained in area
        df_words_contained = df_words[df_words["int_area"] / df_words["w_area"] >= 0.75]

        # Group text by parent
        df_text_parent = (df_words_contained.groupby('parent')
                          .agg(x1=("x1", np.min),
                               x2=("x2", np.max),
                               y1=("y1", np.min),
                               y2=("y2", np.max),
                               value=("value", lambda x: ' '.join(x)))
                          .sort_values(by=["y1", "x1"])
                          )

        # Concatenate all lines
        return df_text_parent["value"].str.cat(sep="\n") or None

    def get_text_table(self, table) -> pd.DataFrame:
        # Get table cells
        table_cells = [[cell for cell in row.items] for row in table.items]

        # Filter dataframe on non empty words
        df_words = self.df[self.df["class"] == "ocrx_word"]
        df_words = df_words[df_words["value"].notnull()]

        # Create dataframe containing all coordinates of Cell objects
        list_cells = list()
        for id_row, row in enumerate(table_cells):
            for id_col, cell in enumerate(row):
                element = {"row": id_row, "col": id_col,
                           "x1_w": cell.x1, "x2_w": cell.x2,
                           "y1_w": cell.y1, "y2_w": cell.y2}
                list_cells.append(element)
        df_cells = pd.DataFrame(list_cells)

        # Cartesian product between two dataframes
        df_word_cells = df_words.merge(df_cells, how="cross")

        # Compute coordinates of intersection
        df_word_cells["x_left"] = df_word_cells[["x1", "x1_w"]].max(axis=1)
        df_word_cells["y_top"] = df_word_cells[["y1", "y1_w"]].max(axis=1)
        df_word_cells["x_right"] = df_word_cells[["x2", "x2_w"]].min(axis=1)
        df_word_cells["y_bottom"] = df_word_cells[["y2", "y2_w"]].min(axis=1)

        # Filter where intersection is not empty
        df_word_cells = df_word_cells[df_word_cells["x_right"] > df_word_cells["x_left"]]
        df_word_cells = df_word_cells[df_word_cells["y_bottom"] > df_word_cells["y_top"]]

        # Compute area of word bbox and intersection
        df_word_cells["w_area"] = (df_word_cells["x2"] - df_word_cells["x1"]) * (df_word_cells["y2"] - df_word_cells["y1"])
        df_word_cells["int_area"] = (df_word_cells["x_right"] - df_word_cells["x_left"]) * (df_word_cells["y_bottom"] - df_word_cells["y_top"])

        # Filter on words where its bbox is contained in area
        df_words_contained = df_word_cells[df_word_cells["int_area"] / df_word_cells["w_area"] >= 0.75]

        # Group text by parent
        df_text_parent = (df_words_contained.groupby(['row', 'col', 'parent'])
                          .agg(x1=("x1", np.min),
                               x2=("x2", np.max),
                               y1=("y1", np.min),
                               y2=("y2", np.max),
                               value=("value", lambda x: ' '.join(x)))
                          .sort_values(by=["row", "col", "y1", "x1"])
                          .groupby(["row", "col"])
                          .agg(text=("value", lambda x: "\n".join(x) or None))
                          .reset_index()
                          )

        # Recreate dataframe from values
        values = [[None] * table.nb_columns for _ in range(table.nb_rows)]
        for rec in df_text_parent.to_dict(orient='records'):
            values[rec.get('row')][rec.get('col')] = rec.get('text')

        return pd.DataFrame(values)

    @classmethod
    def of(cls, image: np.ndarray, lang: str) -> "OCRPage":
        # Preprocess for OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hocr_html = pytesseract.image_to_pdf_or_hocr(gray,
                                                     extension="hocr",
                                                     config="--psm 1",
                                                     lang=lang).decode('utf-8')

        return cls(hocr_html=hocr_html)
