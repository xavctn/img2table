# coding: utf-8
import json
import os
import pickle
import subprocess
from typing import NamedTuple, Dict

import azure
import boto3
import pytest
import requests
from google.cloud import vision

from tests import MOCK_DIR


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def mock_tesseract(monkeypatch):
    def mockreturn(*args, **kwargs):
        with open(os.path.join(MOCK_DIR, "tesseract_hocr.html"), "r") as f:
            return f.read().encode("utf-8")

    monkeypatch.setattr(subprocess, "check_output", mockreturn)


@pytest.fixture
def mock_vision(monkeypatch):
    class MockPost:
        def json(self, *args, **kwargs):
            with open(os.path.join(MOCK_DIR, "vision.json"), "r") as f:
                return json.load(f)

    def mock_post(*args, **kwargs):
        return MockPost()

    # Mock post to API
    monkeypatch.setattr(requests, "post", mock_post)

    def mock_init(*args, **kwargs):
        pass

    def mock_annotate(*args, **kwargs):
        with open(os.path.join(MOCK_DIR, "vision.pkl"), "rb") as f:
            resp = pickle.load(f)

        return resp

    # Mock Vision API annotate
    monkeypatch.setattr(vision.ImageAnnotatorClient, "__init__", mock_init)
    monkeypatch.setattr(vision.ImageAnnotatorClient, "batch_annotate_images", mock_annotate)


@pytest.fixture
def mock_textract(monkeypatch):
    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def detect_document_text(*args, **kwargs):
            with open(os.path.join(MOCK_DIR, "textract.json"), "r") as f:
                resp = json.load(f)

            return resp

    # Mock boto3 client
    monkeypatch.setattr(boto3, "client", MockClient)


@pytest.fixture
def mock_azure(monkeypatch):
    class MockRead(NamedTuple):
        headers: Dict

    def mock_read_in_stream(*args, **kwargs):
        return MockRead(headers={"Operation-Location": "zz/zz"})

    def mock_get_read_result(*args, **kwargs):
        with open(os.path.join(MOCK_DIR, "azure.pkl"), "rb") as f:
            resp = pickle.load(f)
        return resp

    # Mock azure client
    monkeypatch.setattr(azure.cognitiveservices.vision.computervision.ComputerVisionClient,
                        "read_in_stream",
                        mock_read_in_stream)
    monkeypatch.setattr(azure.cognitiveservices.vision.computervision.ComputerVisionClient,
                        "get_read_result",
                        mock_get_read_result)

