"""Profile class inspired by pyccl."""

import numpy as np


class Profile(object):
    def __init__(self):
        pass

    def __call__(self, lmax):
        pass


class ProfileBeta(Profile):
    def __init__(self, theta_c, beta=1, n=100, num_points=3):
        """
        Projected beta profile.

        Parameters
        ----------
        theta_c: float
            Core radius in arcmins.
        beta: float
            Beta parameter of the profile.
        n: float
            Maximum theta in units of theta_c.
        num_points: int
            Number of samples per theta_c.
        """
        self.theta = np.linspace(0, max(n * theta_c, 5 * 60), int(num_points * n))
        self.p = self.beta_profile(self.theta, theta_c, beta=beta, y0=1)

    @staticmethod
    def beta_profile(r, theta_c, beta, y0=1):
        return y0 * (1 + (r / theta_c) ** 2) ** (-0.5 * (3 * beta - 1))

    def __call__(self, lmax):
        import healpy as hp

        u = hp.beam2bl(self.p, np.deg2rad(self.theta / 60), lmax=lmax)
        u /= u[0]
        return u


class ProfileGaussian(Profile):
    def __init__(self, fwhm):
        """
        Gaussian profile.

        Parameters
        ----------
        fwhm : float
            Full width half max in arcmins.
        """
        self.fwhm_rad = np.deg2rad(fwhm / 60)

    def __call__(self, lmax):
        """Compute Fourier space directly."""
        sigma = self.fwhm_rad / (np.sqrt(8 * np.log(2)))
        ell = np.arange(lmax + 1)
        return np.exp(-0.5 * ell * (ell + 1) * sigma**2)
