from setuptools import find_packages, setup

with open("./requirements.txt", "r") as lines:
    requirements = lines.read().strip().splitlines()

with open("./README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="img2table",
    version="0.0.1",
    author="Xavier Canton",
    description="img2table is a table identification and extraction Python Library based on OpenCV image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xavctn/img2table",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
)
