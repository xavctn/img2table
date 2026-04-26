import json
from pathlib import Path

from img2table.document import Image
from img2table.ocr import TextractOCR
from tests import MOCK_DIR
from tests.ocr_data_utils import read_ocr_data


def test_map_response(mock_textract) -> None:  # noqa: ANN001, ARG001
    img = Image(src="test_data/test.png")

    with (Path(MOCK_DIR) / "textract.json").open() as f:
        resp = json.load(f)

    result = TextractOCR().map_response(response=resp, image=next(iter(img.images)))

    with (Path("test_data") / "content.json").open() as f:
        expected = json.load(f)

    assert result == expected


def test_textract_ocr(mock_textract) -> None:  # noqa: ANN001, ARG001
    img = Image(src="test_data/test.png")
    ocr = TextractOCR(
        aws_access_key_id="aws_access_key_id",
        aws_secret_access_key="aws_secret_access_key",
        aws_session_token="aws_session_token",
        region="eu-west-1",
    )

    result = ocr.of(document=img)

    expected = read_ocr_data("test_data/ocr.csv")

    assert result == expected
