import os,glob
import numpy as np
import distutils
import subprocess
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('healqest',parent_package,top_path)
    config.add_extension('src.cwignerd', ['healqest/src/wignerd.pyf', 'healqest/src/wignerd.c'])
    return config


class BuildF2Py(build_ext):
    def run(self):
        F2PY = os.environ.get("F2PY", "f2py")
        # Compile with f2py
        subprocess.run(
            [
                F2PY,
                "-c",
                "-m",
                "healqest.src.cwignerd",
                "healqest/src/wignerd.pyf",
                "healqest/src/wignerd.c",
            ],
            check=True,
        )

        outdir = os.path.join("healqest", "src")
        subprocess.run(
            f"mv cwignerd*.so {outdir}",
            shell=True,
        )


if __name__ == "__main__":
    try:
        from numpy.distutils.core import setup
        packages = []
        setup(packages=packages, configuration=configuration)
    except Exception:
        # alternative build for py>3.12 or numpy>1.23
        # using setuptools + meson based f2py.
        from setuptools import setup
        setup(
            name="healqest",
            packages=["healqest", "healqest.src"],
            cmdclass={"build_ext": BuildF2Py}
        )
