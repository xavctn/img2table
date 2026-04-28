from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def iter_extensions() -> list[Extension]:
    pyx_files = sorted(Path("src").rglob("*.pyx"))
    include_dirs = [np.get_include()]
    define_macros = [("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")]

    return [
        Extension(
            name=".".join(pyx_file.with_suffix("").parts[1:]),
            sources=[pyx_file.as_posix()],
            include_dirs=include_dirs,
            define_macros=define_macros,
        )
        for pyx_file in pyx_files
    ]


setup(
    ext_modules=cythonize(
        iter_extensions(),
        compiler_directives={"language_level": "3", "linetrace": True},
    )
)
