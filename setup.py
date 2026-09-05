import os
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


wignerd_ext = Extension(
    "healqest.cwignerd",
    sources=["healqest/cython/wignerd.pyx", "healqest/src/wignerd.c"],
    include_dirs=[os.path.join("healqest", "src"), np.get_include()],
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)


if __name__ == "__main__":
    setup(
        name="healqest",
        version="0.1.0",
        packages=find_packages(),
        include_package_data=True,
        scripts=["scripts/make_ilc.py"],
        ext_modules=cythonize(
            wignerd_ext,
            compiler_directives={
                'language_level': "3",
                'boundscheck': False,
                'wraparound': False,
                'initializedcheck': True,
            },
        ),
        package_data={"healqest": ["*.so", "src/*.h", "data/*", "data/camb/*"]},
    )
