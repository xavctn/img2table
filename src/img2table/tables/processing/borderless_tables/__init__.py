from typing import Optional

import numpy as np
import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.columns import identify_columns
from img2table.tables.processing.borderless_tables.layout import segment_image
from img2table.tables.processing.borderless_tables.rows import identify_delimiter_group_rows
from img2table.tables.processing.borderless_tables.table import identify_table
from img2table.tables.processing.common import is_contained_cell


def coherent_table(tb: Table, elements: list[Cell]) -> Optional[Table]:
    """
    Check coherency of top/bottom part of the table
    :param tb: table
    :param elements: list of elements
    :return: resized table if relevant
    """
    # Dataframe of rows
    df_rows = pl.DataFrame([{"row_id": row_id, "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2}
                            for row_id, row in enumerate(tb.items)
                            for c in row.items])
    # Dataframe of elements
    df_elements = pl.DataFrame([{"x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2} for c in elements])

    df_relevant_rows = (df_rows.unique()
                        .with_columns(pl.col("x1").len().over("row_id").alias("nb_cells"))
                        .filter(pl.col("nb_cells") >= 3)
                        )

    if df_relevant_rows.height == 0:
        return None

    # Get elements in each cells and identify coherent rows
    rel_rows = (df_relevant_rows.join(df_elements, how="cross")
                .with_columns(x_overlap=pl.min_horizontal("x2", "x2_right") - pl.max_horizontal("x1", "x1_right"),
                              y_overlap=pl.min_horizontal("y2", "y2_right") - pl.max_horizontal("y1", "y1_right"),
                              area=(pl.col("x2_right") - pl.col("x1_right")) * (pl.col("y2_right") - pl.col("y1_right")))
                .filter(pl.col("x_overlap") > 0, pl.col("y_overlap") > 0)
                .with_columns(area_overlap=pl.col("x_overlap") * pl.col("y_overlap"))
                .filter(pl.col("area_overlap") / pl.col("area") >= 0.5)
                .group_by("row_id").len()
                .filter(pl.col("len") > 1)
                .select(pl.col("row_id").min().alias("min_row"), pl.col("row_id").max().alias("max_row"))
                .to_dicts()
                )

    if len(rel_rows) > 0 and rel_rows[0].get("min_row") is not None and rel_rows[0].get("max_row") is not None:
        # Get new rows
        new_rows = tb.items[rel_rows[0].get("min_row"):rel_rows[0].get("max_row") + 1]
        if len(new_rows) >= 2:
            return Table(rows=new_rows, borderless=True)

    return None


def deduplicate_tables(identified_tables: list[Table], existing_tables: list[Table]) -> list[Table]:
    """
    Deduplicate identified borderless tables with already identified tables in order to avoid duplicates and overlap
    :param identified_tables: list of borderless tables identified
    :param existing_tables: list of already identified tables
    :return: deduplicated list of identified borderless tables
    """
    # Sort tables by area
    identified_tables = sorted(identified_tables, key=lambda tb: tb.area, reverse=True)

    # For each table check if it does not overlap with an existing table
    final_tables = []
    for table in identified_tables:
        if not any(max(is_contained_cell(inner_cell=table.cell, outer_cell=tb.cell, percentage=0.1),
                       is_contained_cell(inner_cell=tb.cell, outer_cell=table.cell, percentage=0.1))
                   for tb in existing_tables + final_tables):
            final_tables.append(table)

    return final_tables


def identify_borderless_tables(thresh: np.ndarray, lines: list[Line], char_length: float, median_line_sep: float,
                               contours: list[Cell], existing_tables: list[Table]) -> list[Table]:
    """
    Identify borderless tables in image
    :param thresh: threshold image array
    :param lines: list of rows detected in image
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param contours: list of image contours
    :param existing_tables: list of detected bordered tables
    :return: list of detected borderless tables
    """
    # Segment image and identify parts that can correspond to tables
    table_segments = segment_image(thresh=thresh,
                                   lines=lines,
                                   char_length=char_length,
                                   median_line_sep=median_line_sep,
                                   existing_tables=existing_tables)

    # In each segment, create groups of rows and identify tables
    tables = []
    for table_segment in table_segments:
        # Identify column groups in segment
        column_group = identify_columns(table_segment=table_segment,
                                        char_length=char_length)

        if column_group:
            # Identify potential table rows
            row_delimiters = identify_delimiter_group_rows(column_group=column_group,
                                                           contours=contours)

            if row_delimiters:
                # Create table from column group and rows
                borderless_table = identify_table(columns=column_group,
                                                  row_delimiters=row_delimiters,
                                                  contours=contours,
                                                  median_line_sep=median_line_sep,
                                                  char_length=char_length)

                if borderless_table:
                    # Check table
                    corrected_table = coherent_table(tb=borderless_table, elements=table_segment.elements)

                    if corrected_table:
                        tables.append(corrected_table)

    return deduplicate_tables(identified_tables=tables,
                              existing_tables=existing_tables)
