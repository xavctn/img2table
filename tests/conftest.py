import json
import pickle
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import azure.cognitiveservices.vision.computervision
import boto3
import pytest
import requests
from _pytest.python_api import ApproxBase, ApproxMapping, ApproxSequenceLike
from google.cloud import vision

from tests import MOCK_DIR


class ApproxBaseReprMixin(ApproxBase):
    def __repr__(self) -> str:
        def recur_repr_helper(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: recur_repr_helper(v) for k, v in obj.items()}
            if isinstance(obj, tuple):
                return tuple(recur_repr_helper(o) for o in obj)
            if isinstance(obj, list):
                return [recur_repr_helper(o) for o in obj]
            return self._approx_scalar(obj)

        return f"approx({recur_repr_helper(self.expected)!r})"


class ApproxNestedSequenceLike(ApproxBaseReprMixin, ApproxSequenceLike):
    def _yield_comparisons(self, actual: Any) -> Iterator[tuple[Any, Any]]:
        for k in range(len(self.expected)):
            if isinstance(self.expected[k], dict):
                mapping = ApproxNestedMapping(
                    self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok
                )
                for el in mapping._yield_comparisons(actual[k]):
                    yield el
            elif isinstance(self.expected[k], (tuple, list)):
                mapping = ApproxNestedSequenceLike(
                    self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok
                )
                for el in mapping._yield_comparisons(actual[k]):
                    yield el
            else:
                yield actual[k], self.expected[k]

    def _check_type(self) -> None:
        pass


class ApproxNestedMapping(ApproxBaseReprMixin, ApproxMapping):
    def _yield_comparisons(self, actual: Mapping[object, Any]) -> Iterator[tuple[Any, Any]]:
        for k in self.expected:
            if isinstance(self.expected[k], dict):
                mapping = ApproxNestedMapping(
                    self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok
                )
                for el in mapping._yield_comparisons(actual[k]):
                    yield el
            elif isinstance(self.expected[k], (tuple, list)):
                mapping = ApproxNestedSequenceLike(
                    self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok
                )
                for el in mapping._yield_comparisons(actual[k]):
                    yield el
            else:
                yield actual[k], self.expected[k]

    def _check_type(self) -> None:
        pass


def nested_approx(
    expected: Any,
    rel: float | None = None,
    abs: float | None = None,  # noqa: A002
    nan_ok: bool = False,
) -> ApproxBase:
    if isinstance(expected, dict):
        return ApproxNestedMapping(expected, rel, abs, nan_ok)
    if isinstance(expected, (tuple, list)):
        return ApproxNestedSequenceLike(expected, rel, abs, nan_ok)
    return pytest.approx(expected, rel, abs, nan_ok)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def mock_tesseract(monkeypatch) -> None:  # noqa: ANN001
    check_output = subprocess.check_output
    run = subprocess.run

    def mock_check_output(*args, **kwargs) -> bytes:  # noqa: ANN002
        if args and "tesseract --list-langs" in args[0]:
            return b"Langs\neng"
        if not args or not str(args[0]).startswith("tesseract "):
            return check_output(*args, **kwargs)
        with (Path(MOCK_DIR) / "tesseract_hocr.html").open() as f:
            return f.read().encode("utf-8")

    def mock_run(*args, **kwargs) -> Any:  # noqa: ANN002
        if not args or args[0] != "tesseract --version":
            return run(*args, **kwargs)

        class MResp:
            @property
            def returncode(self) -> int:
                return 0

        return MResp()

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    monkeypatch.setattr(subprocess, "run", mock_run)


@pytest.fixture
def mock_vision(monkeypatch) -> None:  # noqa: ANN001
    class MockPost:
        def json(self, *args, **kwargs) -> dict:  # noqa: ANN002, ARG002
            with (Path(MOCK_DIR) / "vision.json").open() as f:
                return json.load(f)

    def mock_post(*args, **kwargs) -> MockPost:  # noqa: ANN002, ARG001
        return MockPost()

    # Mock post to API
    monkeypatch.setattr(requests, "post", mock_post)

    def mock_init(*args, **kwargs) -> None:  # noqa: ANN002
        pass

    def mock_annotate(*args, **kwargs) -> Any:  # noqa: ANN002, ARG001
        with (Path(MOCK_DIR) / "vision.pkl").open("rb") as f:
            return pickle.load(f)

    # Mock Vision API annotate
    monkeypatch.setattr(vision.ImageAnnotatorClient, "__init__", mock_init)
    monkeypatch.setattr(vision.ImageAnnotatorClient, "batch_annotate_images", mock_annotate)


@pytest.fixture
def mock_textract(monkeypatch) -> None:  # noqa: ANN001
    class MockClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002
            pass

        def detect_document_text(*args, **kwargs) -> dict:  # noqa: ANN002, ARG002
            with (Path(MOCK_DIR) / "textract.json").open() as f:
                return json.load(f)

    # Mock boto3 client
    monkeypatch.setattr(boto3, "client", MockClient)


@pytest.fixture
def mock_azure(monkeypatch) -> None:  # noqa: ANN001
    class MockRead(NamedTuple):
        headers: dict

    def mock_read_in_stream(*args, **kwargs) -> MockRead:  # noqa: ANN002, ARG001
        return MockRead(headers={"Operation-Location": "zz/zz"})

    def mock_get_read_result(*args, **kwargs) -> Any:  # noqa: ANN002, ARG001
        with (Path(MOCK_DIR) / "azure.pkl").open("rb") as f:
            return pickle.load(f)

    # Mock azure client
    monkeypatch.setattr(
        azure.cognitiveservices.vision.computervision.ComputerVisionClient,
        "read_in_stream",
        mock_read_in_stream,
    )
    monkeypatch.setattr(
        azure.cognitiveservices.vision.computervision.ComputerVisionClient,
        "get_read_result",
        mock_get_read_result,
    )


@pytest.fixture
def mock_surya(monkeypatch) -> None:  # noqa: ANN001
    def mock_run_ocr(*args, **kwargs) -> Any:  # noqa: ANN002, ARG001
        with (Path(MOCK_DIR) / "surya.pkl").open("rb") as f:
            return pickle.load(f)

    import surya.recognition

    # Mock surya
    monkeypatch.setattr(surya.recognition.RecognitionPredictor, "__call__", mock_run_ocr)
