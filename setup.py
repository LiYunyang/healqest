import os,glob
import numpy as np
import distutils
from distutils.extension import Extension

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('healqest',parent_package,top_path)
    config.add_extension('src.cwignerd', ['healqest/src/wignerd.pyf', 'healqest/src/wignerd.c'])
    return config


#fftlog_module = Extension( name="cosmotools3.ext.pycl2xi.pycl2xi._fftlog",
#                           sources=["cosmotools3/ext/pycl2xi/src/fftlog.c"],
#                           depends=["cosmotools3/ext/pycl2xi/src/fftlog.h"],
#                           libraries=["fftw3"],
#                           extra_compile_args=["-std=gnu99"]
#                           )

if __name__ == "__main__":
    from numpy.distutils.core import setup

    packages     = []#["cosmotools3","cosmotools3.ext","cosmotools3.ext.quicklens","cosmotools3.ext.pycl2xi","cosmotools3.ext.pycl2xi.pycl2xi","cosmotools3.ext.pycl2xi.src"]

    #data_dir     = os.path.join("data","cl","planck_wp_highL","*")
    #package_data = { "cosmotools3" : [data_dir] }

    setup(packages      = packages,
          #package_data  = package_data,
          #ext_modules   = [fftlog_module],
          configuration = configuration)
