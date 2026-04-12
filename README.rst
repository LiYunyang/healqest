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

If you want to install the package, you can do it by running::

    python setup.py build_ext # (optional for analytic response calculation)
    pip install [-e] ./

Currently, ``build_ext`` is not working with ``pip``, so one has to compile it manually before installation.

Development
------------

After cloning, install the pre-commit hooks::

    pre-commit install

This runs ``ruff`` (lint + format) and basic file checks on every commit.
