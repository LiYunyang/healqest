import os,glob
import numpy as np
import distutils
from distutils.extension import Extension

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('healqest',parent_package,top_path)
    config.add_extension('src.cwignerd', ['healqest/src/wignerd.pyf', 'healqest/src/wignerd.c'])
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    packages     = []

    setup(packages = packages, configuration = configuration)
