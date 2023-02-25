# coding: utf-8

import os
import subprocess

CWD = os.path.dirname(__file__)
MOCK_DIR = os.path.join(CWD, "_mock_data")

TESSERACT_INSTALL = subprocess.run("tesseract --version", shell=True).returncode == 0
