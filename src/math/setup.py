import os,glob
import numpy as np
import distutils

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('quicklens',parent_package,top_path)
    config.add_extension('cwignerd', ['wignerd.pyf', 'wignerd.c'])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    packages = []

    setup(packages=packages,configuration=configuration)
