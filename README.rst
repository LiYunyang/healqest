HEALQest
------------
This is the core part of HEALPix based quadratic estimator.

Requirements
------------

* `Python <http://www.python.org>`_ 3.7, 3.8, 3.9, or 3.10

* `Numpy <http://numpy.scipy.org/>`_ (tested with version >=1.5.0)

* `Matplotlib <http://matplotlib.sourceforge.net/>`_

* `Astropy <http://www.astropy.org>`_

* `CAMB <https://github.com/cmbant/CAMB>`_

* `Healpy <https://github.com/healpy/healpy>`_

Install
------------
If you want to install the package, you can do it by running:
    ```bash
    python setup.py build_ext --inplace
    pip install [-e] ./
    ```
Currently, the build_ext is not working with `pip`, so one has to compile it manually before installation.


Or if you prefer the the old way, the only thing you need to do is either add the following line at the beginning of your code::

    sys.path.append("path_to_this_directory/healqest/")
    
or on a terminal do::

    export PYTHON_PATH=${PYTHON_PATH}:PATH_TO_THIS_DIRECTORY/healqest/

Note that the path has changed (no more `src` directory) to make the package installable as `healqest`.

You need to compile the cython code if you want to compute the analytic response function. This can be done by simply running::

    python setup.py build_ext --inplace


Example
------------
See example.py.
