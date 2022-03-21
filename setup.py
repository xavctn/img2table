import setuptools

setuptools.setup(
    name="img2table",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
