# coding: utf-8
import os
import subprocess

import pytest

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



