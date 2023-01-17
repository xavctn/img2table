# coding: utf-8
import os
import pickle

import pandas as pd
import pytest

from img2table.document import Image
from img2table.ocr import AzureOCR
from img2table.ocr.data import OCRDataframe
from tests import MOCK_DIR


def test_content(mock_azure):
    img = Image("test_data/test.png")
    ocr = AzureOCR(endpoint="aa", api_key="bb")

    result = ocr.content(document=img)

    with open(os.path.join(MOCK_DIR, "azure.pkl"), "rb") as f:
        expected = pickle.load(f)

    assert list(result) == [expected]


def test_to_ocr_df(mock_azure):
    ocr = AzureOCR(endpoint="aa", api_key="bb")
    with open(os.path.join(MOCK_DIR, "azure.pkl"), "rb") as f:
        content = pickle.load(f)

    result = ocr.to_ocr_dataframe(content=[content])

    expected = OCRDataframe(df=pd.read_csv("test_data/ocr_df.csv", sep=";"))

    assert result == expected


def test_azure_ocr(mock_azure):
    # Test init error
    with pytest.raises(TypeError) as e_info:
        AzureOCR(api_key=8, endpoint="a")

    with pytest.raises(TypeError) as e_info:
        AzureOCR(api_key="a", endpoint=0)

    with pytest.raises(ValueError) as e_info:
        AzureOCR(api_key="a")

    with pytest.raises(ValueError) as e_info:
        AzureOCR(endpoint="a")

    img = Image("test_data/test.png")
    ocr = AzureOCR(endpoint="aa", api_key="bb")

    result = ocr.of(document=img)

    expected = OCRDataframe(df=pd.read_csv("test_data/ocr_df.csv", sep=";"))

    assert result == expected
