from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def get_pyx_extensions() -> list[Extension]:
    return [
        Extension(
            name=".".join(pyx_file.with_suffix("").parts[1:]),
            sources=[pyx_file.as_posix()],
            include_dirs=[np.get_include()],
        )
        for pyx_file in sorted(Path("src").rglob("*.pyx"))
    ]


setup(ext_modules=cythonize(get_pyx_extensions(), compiler_directives={"language_level": "3"}))
