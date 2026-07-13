"""Wigner-d transforms evaluated with Gauss-Legendre quadrature."""

import numpy as np

from healqest import cwignerd


class GaussLegendreQuadrature:
    """Gauss-Legendre quadrature for Wigner-d transforms.

    The implementation is adapted from ``libkms_ist`` by Kendrick Smith.

    Parameters
    ----------
    npoints : int
        Number of quadrature points.

    Attributes
    ----------
    npoints : int
        Number of quadrature points.
    zvec : numpy.ndarray
        Quadrature nodes in the interval ``[-1, 1]``.
    wvec : numpy.ndarray
        Integration weights for the quadrature nodes.
    """

    def __init__(self, npoints):
        """Initialize quadrature nodes and weights.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.
        """
        self.npoints = npoints
        self.zvec, self.wvec = cwignerd.init_gauss_legendre_quadrature(npoints)

    def cf_from_cl(self, s1, s2, cl):
        r"""Evaluate a correlation function from harmonic coefficients.

        This computes ``cf[j] = sum_l cl[l] d^l_{s1 s2}(zvec[j])``.

        Parameters
        ----------
        s1, s2 : int
            Spin indices of the Wigner-d matrix.
        cl : np.ndarray
            Harmonic coefficients indexed by multipole ``l``.

        Returns
        -------
        np.ndarray
            Samples of the correlation function at ``self.zvec``.
        """
        lmax = len(cl) - 1

        if np.iscomplexobj(cl):
            cl2d = np.concatenate([cl.real, cl.imag])
            output = cwignerd.wignerd_cf_from_cl(s1, s2, 2, self.npoints, lmax, self.zvec, cl2d).reshape(
                2, -1
            )
            return output[0] + 1j * output[1]

        return cwignerd.wignerd_cf_from_cl(s1, s2, 1, self.npoints, lmax, self.zvec, cl)

    def cl_from_cf(self, lmax, s1, s2, cf):
        r"""Integrate sampled correlation values into harmonic coefficients.

        This computes the Gauss-Legendre approximation to
        ``cl[l] = integral cf(x) d^l_{s1 s2}(x) dx`` using ``self.zvec``
        and ``self.wvec``. The input ``cf[j]`` should represent a polynomial
        sampled at ``x = self.zvec[j]`` with degree less than
        ``self.npoints - lmax``.

        Parameters
        ----------
        lmax : int
            Maximum multipole to compute.
        s1, s2 : int
            Spin indices of the Wigner-d matrix.
        cf : np.ndarray
            Correlation-function samples at ``self.zvec``.

        Returns
        -------
        np.ndarray
            Harmonic coefficients up to ``lmax``.
        """
        if np.iscomplexobj(cf):
            cf2d = np.concatenate([cf.real, cf.imag])
            output = cwignerd.wignerd_cl_from_cf(
                s1, s2, 2, self.npoints, lmax, self.zvec, self.wvec, cf2d
            ).reshape(2, -1)
            return output[0] + 1j * output[1]

        return cwignerd.wignerd_cl_from_cf(s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf)
