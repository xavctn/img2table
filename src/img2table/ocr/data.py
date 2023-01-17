# coding: utf-8
from dataclasses import dataclass

import numpy as np
import pandas as pd

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table


@dataclass
class OCRDataframe:
    df: pd.DataFrame

    def page(self, page_number: int = 0) -> "OCRDataframe":
        return OCRDataframe(df=self.df[self.df["page"] == page_number])

    def get_text_cell(self, cell: Cell, margin: int = 0, page_number: int = None, min_confidence: int = 50) -> str:
        """
        Get text corresponding to cell
        :param cell: Cell object in document
        :param margin: margin to take around cell
        :param page_number: page number of the cell
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: text contained in cell
        """
        # Define relevant bounding box
        bbox = cell.bbox(margin=margin)

        # Filter dataframe on relevant page
        df_words = self.df[(self.df["class"] == "ocrx_word")]
        if page_number:
            df_words = df_words[df_words["page"] == page_number]
        # Filter dataframe on relevant words
        df_words = df_words[df_words["value"].notnull() & (df_words["confidence"] >= min_confidence)]

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

        # Group text by parents
        df_text_parent = (df_words_contained.groupby('parent')
                          .agg(x1=("x1", np.min),
                               x2=("x2", np.max),
                               y1=("y1", np.min),
                               y2=("y2", np.max),
                               value=("value", lambda x: ' '.join(x)))
                          .sort_values(by=["y1", "x1"])
                          )

        # Concatenate all lines
        return df_text_parent["value"].astype(str).str.cat(sep="\n").strip() or None

    def get_text_table(self, table: Table, page_number: int = None, min_confidence: int = 50) -> Table:
        """
        Identify text located in Table object
        :param table: Table object
        :param page_number: page number of the cell
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: table with content set on all cells
        """
        # Filter dataframe on relevant page
        df_words = self.df[(self.df["class"] == "ocrx_word")]
        if page_number:
            df_words = df_words[df_words["page"] == page_number]
        # Filter dataframe on relevant words
        df_words = df_words[df_words["value"].notnull() & (df_words["confidence"] >= min_confidence)]

        # Create dataframe containing all coordinates of Cell objects
        list_cells = [{"row": id_row, "col": id_col, "x1_w": cell.x1, "x2_w": cell.x2, "y1_w": cell.y1, "y2_w": cell.y2}
                      for id_row, row in enumerate(table.items)
                      for id_col, cell in enumerate(row.items)]
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

        # If no words are contained, return the table
        if len(df_words_contained) == 0:
            return table

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

        # Implement found values to table cells content
        for rec in df_text_parent.to_dict(orient='records'):
            table.items[rec.get('row')].items[rec.get('col')].content = rec.get('text').strip() or None

        return table

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            try:
                assert self.df.equals(other.df)
                return True
            except AssertionError:
                return False
        return False
