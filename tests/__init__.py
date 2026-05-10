import subprocess
from pathlib import Path

CWD = Path(__file__).parent
MOCK_DIR = CWD / "_mock_data"

TESSERACT_INSTALL = subprocess.run("tesseract --version", shell=True, check=False).returncode == 0  # noqa: S602, S607
