# coding: utf-8
from dataclasses import dataclass

import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table


@dataclass
class OCRDataframe:
    df: pl.LazyFrame

    def page(self, page_number: int = 0) -> "OCRDataframe":
        # Filter dataframe on specific page
        df_page = self.df.filter(pl.col('page') == page_number)
        return OCRDataframe(df=df_page)

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
        df_words = self.df.filter(pl.col('class') == "ocrx_word")
        if page_number:
            df_words = df_words.filter(pl.col('page') == page_number)
        # Filter dataframe on relevant words
        df_words = df_words.filter(pl.col('value').is_not_null() & (pl.col('confidence') >= min_confidence))

        # Compute coordinates of intersection
        df_words = (df_words.with_columns([pl.lit(bbox[0]).alias('x1_bbox'),
                                           pl.lit(bbox[1]).alias('y1_bbox'),
                                           pl.lit(bbox[2]).alias('x2_bbox'),
                                           pl.lit(bbox[3]).alias('y2_bbox')]
                                          )
                    .with_columns([pl.max([pl.col('x1'), pl.col('x1_bbox')]).alias('x_left'),
                                   pl.max([pl.col('y1'), pl.col('y1_bbox')]).alias('y_top'),
                                   pl.min([pl.col('x2'), pl.col('x2_bbox')]).alias('x_right'),
                                   pl.min([pl.col('y2'), pl.col('y2_bbox')]).alias('y_bottom'),
                                   ])
                    )

        # Filter where intersection is not empty
        df_intersection = (df_words.filter(pl.col("x_right") > pl.col("x_left"))
                           .filter(pl.col("y_bottom") > pl.col("y_top"))
                           )

        # Compute area of word bbox and intersection
        df_areas = (df_intersection.with_columns([
            ((pl.col('x2') - pl.col('x1')) * (pl.col('y2') - pl.col('y1'))).alias('w_area'),
            ((pl.col('x_right') - pl.col('x_left')) * (pl.col('y_bottom') - pl.col('y_top'))).alias('int_area')
        ])
        )

        # Filter on words where its bbox is contained in area
        df_words_contained = df_areas.filter(pl.col('int_area') / pl.col('w_area') >= 0.75)

        # Group text by parents
        df_text_parent = (df_words_contained
                          .groupby('parent')
                          .agg([pl.col('x1').min(),
                                pl.col('x2').max(),
                                pl.col('y1').min(),
                                pl.col('y2').max(),
                                pl.col('value').list()])
                          .sort([pl.col("y1"), pl.col("x1")])
                          )

        # Concatenate all lines
        text_lines = (df_text_parent.select(pl.col('value'))
                      .collect()
                      .get_column('value')
                      .to_list()
                      )

        return "\n".join([" ".join(line).strip() for line in text_lines]).strip() or None

    def get_text_table(self, table: Table, page_number: int = None, min_confidence: int = 50) -> Table:
        """
        Identify text located in Table object
        :param table: Table object
        :param page_number: page number of the cell
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: table with content set on all cells
        """
        # Filter dataframe on relevant page
        df_words = self.df.filter(pl.col('class') == "ocrx_word")
        if page_number:
            df_words = df_words.filter(pl.col('page') == page_number)
        # Filter dataframe on relevant words
        df_words = df_words.filter(pl.col('value').is_not_null() & (pl.col('confidence') >= min_confidence))

        # Create dataframe containing all coordinates of Cell objects
        list_cells = [{"row": id_row, "col": id_col, "x1_w": cell.x1, "x2_w": cell.x2, "y1_w": cell.y1, "y2_w": cell.y2}
                      for id_row, row in enumerate(table.items)
                      for id_col, cell in enumerate(row.items)]
        df_cells = pl.from_dicts(dicts=list_cells).lazy()

        # Cartesian product between two dataframes
        df_word_cells = df_words.join(other=df_cells, how="cross")

        # Compute coordinates of intersection
        df_word_cells = df_word_cells.with_columns([pl.max([pl.col('x1'), pl.col('x1_w')]).alias('x_left'),
                                                    pl.max([pl.col('y1'), pl.col('y1_w')]).alias('y_top'),
                                                    pl.min([pl.col('x2'), pl.col('x2_w')]).alias('x_right'),
                                                    pl.min([pl.col('y2'), pl.col('y2_w')]).alias('y_bottom'),
                                                    ])

        # Filter where intersection is not empty
        df_intersection = (df_word_cells.filter(pl.col("x_right") > pl.col("x_left"))
                           .filter(pl.col("y_bottom") > pl.col("y_top"))
                           )

        # Compute area of word bbox and intersection
        df_areas = (df_intersection.with_columns([
            ((pl.col('x2') - pl.col('x1')) * (pl.col('y2') - pl.col('y1'))).alias('w_area'),
            ((pl.col('x_right') - pl.col('x_left')) * (pl.col('y_bottom') - pl.col('y_top'))).alias('int_area')
        ])
        )

        # Filter on words where its bbox is contained in area
        df_words_contained = df_areas.filter(pl.col('int_area') / pl.col('w_area') >= 0.75)

        # Group text by parent
        df_text_parent = (df_words_contained
                          .groupby(['row', 'col', 'parent'])
                          .agg([pl.col('x1').min(),
                                pl.col('x2').max(),
                                pl.col('y1').min(),
                                pl.col('y2').max(),
                                pl.col('value').list().apply(lambda x: ' '.join(x)).alias('value')])
                          .sort([pl.col("row"), pl.col("col"), pl.col('y1'), pl.col('x1')])
                          .groupby(['row', 'col'])
                          .agg(pl.col('value').list().apply(lambda x: '\n'.join(x).strip()).alias('text'))
                          )

        # Implement found values to table cells content
        for rec in df_text_parent.collect().to_dicts():
            table.items[rec.get('row')].items[rec.get('col')].content = rec.get('text') or None

        return table

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            try:
                assert self.df.collect().frame_equal(other.df.collect())
                return True
            except AssertionError:
                return False
        return False
