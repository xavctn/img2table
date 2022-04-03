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
