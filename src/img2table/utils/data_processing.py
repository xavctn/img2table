# coding: utf-8

import copy
from typing import Dict, Union, List

import pandas as pd


def is_empty_row(row: Dict, columns: Union[List, pd.Index]) -> bool:
    """
    Check if dataframe row is empty
    :param row: dataframe row
    :param columns: dataframe columns
    :return: boolean indicating if row is empty
    """
    for col in columns:
        if row[col] == row[col] and row[col] is not None:
            return True
    return False


def remove_empty_rows(table: pd.DataFrame) -> pd.DataFrame:
    """
    Remove empty rows from dataframe
    :param table: pandas dataframe
    :return: dataframe with no empty rows
    """
    # Get original columns
    orig_columns = table.columns

    # Check if rows are empty and remove those rows
    table['is_not_empty'] = table.apply(lambda row: is_empty_row(row, orig_columns), axis=1)
    table = table[table['is_not_empty']][orig_columns]

    return table

def split_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split rows in multiple rows if necessary
    :param df: dataframe
    :return: dataframe with splitted rows
    """
    new_rows = list()
    for row in df.to_dict('records'):
        # Check splits by newline
        splits = [len(str(v).split('\n')) for k, v in row.items() if v is not None]
        # If splits are consistent, create multiple rows
        if len(splits) > 1 and len(set(splits)) == 1 and splits[0] > 1:
            nb_splits = list(splits)[0]
            row_splitted = [{k: v.split('\n')[idx] or None if v is not None else None for k, v in row.items()} for idx
                            in range(nb_splits)]
            new_rows = new_rows + row_splitted
        else:
            new_rows.append(row)

    return pd.DataFrame(new_rows)


def remove_empty_columns(table: pd.DataFrame) -> pd.DataFrame:
    """
    Remove empty columns from dataframe
    :param table: pandas dataframe
    :return: dataframe with no empty columns
    """
    cols = list(table.columns)
    orig_cols = copy.deepcopy(cols)

    for col in cols:
        distinct_values = list(set(list(table[col].values)))
        if len(distinct_values) == 1 and distinct_values[0] is None:
            orig_cols.remove(col)

    return table[orig_cols]
