import os
import distutils
import subprocess
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages


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

        outdir = os.path.join("healqest", )
        subprocess.run(
            f"mv cwignerd*.so {outdir}",
            shell=True,
        )


if __name__ == "__main__":
    # the old setup, only compile the wigner code.
    # from numpy.distutils.core import setup
    # packages = []
    # setup(packages=packages, configuration=configuration)

    # full-functioning setup, compile the wigner code and install the package.
    # tested for py>=3.12 or numpy>=1.23
    # using setuptools + meson based f2py.
    setup(
        name="healqest",
        version="0.1.0",
        packages=find_packages(),
        include_package_data=True,
        package_data={"healqest": ["camb/*", "*.so"]},
        cmdclass={"build_ext": BuildF2Py},
        setup_requires=[
            # "ducc0"
        ],
    )
