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
-----------

Code style
~~~~~~~~~~
Currently, the code format is enforced by ``ruff``, with customized configuration in ``pyproject.toml``.
``ruff`` is good at auto-formatting the code with minimal ambiguity, but there are cases where the strict
formatting may not be desirable (compromising readability for example). In such cases, we disable the
formatting on a case-by-case basis. The comon cases are:

- Long line of arguments: By default, ``ruff`` will break the arguments into individual lines that may hurt readability, especially in scripts code where the same pattern might repeat. It is okay to add ``# fmt: off`` in this cases. Note that, this will also disable other formatting rules within the line, so it should be used with caution.
- Complex structure. It would be good to fix these eventually, but one can also disable this rule by ``noqa: C901`` if it is too much work to fix it right now.


To get all these fully automated, install the pre-commit hooks::

    pre-commit install

This runs ``ruff`` (lint + format) and basic file checks on every commit.
