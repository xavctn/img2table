# coding: utf-8

from typing import List, Union

import polars as pl

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import ImageSegment, ColumnGroup, Whitespace


def get_whitespaces(segment: Union[ImageSegment, ColumnGroup], vertical: bool = True, min_width: float = 0,
                    min_height: float = 1, pct: float = 0.25, continuous: bool = True) -> List[Whitespace]:
    """
    Identify whitespaces in segment
    :param segment: image segment
    :param vertical: boolean indicating if vertical or horizontal whitespaces are identified
    :param min_width: minimum width of the detected whitespaces
    :param min_height: minimum height of the detected whitespaces
    :param pct: minimum percentage of the segment height/width to account for a whitespace
    :param continuous: boolean indicating if only continuous whitespaces are retrieved
    :return: list of vertical or horizontal whitespaces
    """
    # Flip object coordinates in horizontal case
    if not vertical:
        flipped_elements = [Cell(x1=el.y1, y1=el.x1, x2=el.y2, y2=el.x2) for el in segment.elements]
        segment = ImageSegment(x1=segment.y1,
                               y1=segment.x1,
                               x2=segment.y2,
                               y2=segment.x2,
                               elements=flipped_elements)

    # Get min/max height of elements in segment
    y_min, y_max = min([el.y1 for el in segment.elements]), max([el.y2 for el in segment.elements])

    # Create dataframe containing elements
    df_elements = pl.concat(
        [pl.LazyFrame([{"x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2} for el in segment.elements]),
         pl.LazyFrame([{"x1": segment.x1, "y1": y, "x2": segment.x2, "y2": y} for y in [y_min, y_max]])]
    )

    # Get dataframe with relevant ranges
    df_x_ranges = (pl.concat([df_elements.select(pl.col('x1').alias('x')), df_elements.select(pl.col('x2').alias('x'))])
                   .unique()
                   .sort(by="x")
                   .select(pl.col('x').alias('x_min'), pl.col('x').shift(-1).alias('x_max'))
                   .filter(pl.col('x_max') - pl.col('x_min') >= min_width)
                   )

    # Get all elements within range and identify whitespaces
    df_elements_ranges = (
        df_x_ranges.join(df_elements, how='cross')
        .filter(pl.min_horizontal('x_max', 'x2') - pl.max_horizontal('x_min', 'x1') > 0)
        .sort(by=["x_min", "x_max", pl.col("y1") + pl.col('y2')])
        .select(pl.col("x_min").alias('x1'),
                pl.col("x_max").alias("x2"),
                pl.col('y2').shift().over("x_min", 'x_max').alias('y1'),
                pl.col('y1').alias('y2')
                )
        .filter(pl.col('y1').is_not_null(),
                pl.col('y2') - pl.col('y1') >= min_height)
    )

    if continuous:
        df_elements_ranges = (df_elements_ranges.filter(pl.col('y2') - pl.col('y1') >= pct * (y_max - y_min))
                              .sort(by=['y1', 'y2', 'x1'])
                              .with_columns(((pl.col('x1') != pl.col('x2').shift())
                                             | (pl.col('y1') != pl.col('y1').shift())
                                             | (pl.col('y2') != pl.col('y2').shift())
                                             ).cast(int).cum_sum().alias('ws_id')
                                            )
                              .group_by("ws_id")
                              .agg(pl.col('x1').min().alias('x1'),
                                   pl.col('y1').min().alias('y1'),
                                   pl.col('x2').max().alias('x2'),
                                   pl.col('y2').max().alias('y2'))
                              .drop("ws_id")
                              .collect()
                              )
        whitespaces = [Whitespace(cells=[Cell(**ws_dict)])
                       for ws_dict in df_elements_ranges.to_dicts()]

    else:
        df_elements_ranges = (
            df_elements_ranges
            .with_columns((pl.col('y2') - pl.col('y1')).sum().over(["x1", "x2"]).alias("ws_height"),
                          (pl.col('y2').max().over(["x1", "x2"]) - pl.col('y2').min().over(["x1", "x2"])).alias("height"),
                          pl.len().over(["x1", "x2"]).alias("nb_ws"))
            .filter(pl.col("ws_height") >= pct * (y_max - y_min),
                    pl.col("ws_height") >= 0.8 * pl.col("height"),
                    (pl.col("nb_ws") == 1) | (pl.col('x2') - pl.col('x1') >= 2 * min_width))
            .drop("ws_height", "height", "nb_ws")
            .collect()
        )

        whitespaces = [Whitespace(cells=[Cell(**ws_dict) for ws_dict in ws_group])
                       for _, ws_group in df_elements_ranges.rows_by_key(key=["x1", "x2"], named=True, include_key=True).items()]

    # Flip object coordinates in horizontal case
    if not vertical:
        whitespaces = [ws.flipped() for ws in whitespaces]

    return whitespaces


def adjacent_whitespaces(w_1: Whitespace, w_2: Whitespace) -> bool:
    """
    Identify if two whitespaces are adjacent
    :param w_1: first whitespace
    :param w_2: second whitespace
    :return: boolean indicating if two whitespaces are adjacent
    """
    x_coherent = len({w_1.x1, w_1.x2}.intersection({w_2.x1, w_2.x2})) > 0
    y_coherent = min(w_1.y2, w_2.y2) - max(w_1.y1, w_2.y1) > 0

    return x_coherent and y_coherent


def identify_coherent_v_whitespaces(v_whitespaces: List[Whitespace]) -> List[Whitespace]:
    """
    From vertical whitespaces, identify the most relevant ones according to height, width and relative positions
    :param v_whitespaces: list of vertical whitespaces
    :return: list of relevant vertical delimiters
    """
    deleted_idx = list()
    for i in range(len(v_whitespaces)):
        for j in range(i, len(v_whitespaces)):
            # Check if both whitespaces are adjacent
            adjacent = adjacent_whitespaces(v_whitespaces[i], v_whitespaces[j])

            if adjacent:
                if v_whitespaces[i].height > v_whitespaces[j].height:
                    deleted_idx.append(j)
                elif v_whitespaces[i].height < v_whitespaces[j].height:
                    deleted_idx.append(i)

    return [ws for idx, ws in enumerate(v_whitespaces) if idx not in deleted_idx]


def deduplicate_whitespaces(ws: List[Whitespace], elements: List[Cell]) -> List[Whitespace]:
    """
    Remove useless whitespaces
    :param ws: list of whitespaces
    :param elements: list of segment elements
    :return: filtered whitespaces
    """
    if len(ws) <= 1:
        return ws

    deleted_idx, merged_ws = list(), list()
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            matching_elements = list()
            for ws_1 in ws[i].cells:
                for ws_2 in ws[j].cells:
                    if min(ws_1.y2, ws_2.y2) - max(ws_1.y1, ws_2.y1) <= 0:
                        continue

                    # Get common area
                    common_area = Cell(x1=min(ws_1.x2, ws_2.x2),
                                       y1=max(ws_1.y1, ws_2.y1),
                                       x2=max(ws_1.x1, ws_2.x1),
                                       y2=min(ws_1.y2, ws_2.y2))

                    # Identify matching elements
                    matching_elements += [el for el in elements if el.x1 >= common_area.x1 and el.x2 <= common_area.x2
                                          and el.y1 >= common_area.y1 and el.y2 <= common_area.y2]

            if len(matching_elements) == 0:
                # Add smallest element to deleted ws
                if ws[i].height > ws[j].height:
                    deleted_idx.append(j)
                elif ws[i].height < ws[j].height:
                    deleted_idx.append(i)
                else:
                    # Create a merged whitespace
                    new_cells = [Cell(x1=min(ws[i].x1, ws[j].x1),
                                      y1=c.y1,
                                      x2=max(ws[i].x2, ws[j].x2),
                                      y2=c.y2)
                                 for c in ws[i].cells + ws[j].cells]
                    merged_ws.append(Whitespace(cells=list(set(new_cells))))
                    deleted_idx += [i, j]

    filtered_ws = [w for idx, w in enumerate(ws) if idx not in deleted_idx]

    # Remove merged whitespaces that are incoherent with filtered whitespaces
    merged_ws = [m_ws for m_ws in merged_ws
                 if not any([min(w.x2, m_ws.x2) - max(w.x1, m_ws.x1) > 0 for w in filtered_ws])]

    if len(merged_ws) > 1:
        # Deduplicate overlapping merged ws
        seq = iter(sorted(merged_ws, key=lambda w: w.area, reverse=True))
        filtered_merged_ws = [next(seq)]
        for w in seq:
            if not any([f_ws for f_ws in filtered_ws if w in f_ws]):
                filtered_merged_ws.append(w)
    else:
        filtered_merged_ws = merged_ws

    return filtered_ws + filtered_merged_ws


def get_relevant_vertical_whitespaces(segment: Union[ImageSegment, ColumnGroup], char_length: float,
                                      median_line_sep: float, pct: float = 0.25) -> List[Whitespace]:
    """
    Identify vertical whitespaces that can be column delimiters
    :param segment: image segment
    :param char_length: average character width in image
    :param median_line_sep: median row separation
    :param pct: minimum percentage of the segment height for a vertical whitespace
    :return: list of vertical whitespaces that can be column delimiters
    """
    # Identify vertical whitespaces
    v_whitespaces = get_whitespaces(segment=segment,
                                    vertical=True,
                                    pct=pct,
                                    min_width=char_length,
                                    min_height=min(median_line_sep, segment.height),
                                    continuous=False)

    # Identify relevant vertical whitespaces that can be column delimiters
    vertical_delims = identify_coherent_v_whitespaces(v_whitespaces=v_whitespaces)

    return deduplicate_whitespaces(ws=vertical_delims, elements=segment.elements)
