"""
Similar to opfilt_teb.py for flatsky, this is for T-only and for healpix maps.

Modified from and built on plancklens/qcinv/opfilt_tt.py
"""

import logging
import numpy as np
import healpy as hp
from functools import cached_property

from . import hp_utils, cinv_utils, opfilt_hp
from .opfilt_hp import alm2map, map2alm
logger = logging.getLogger(__name__)


def calc_prep(maps, s_inv_filt, n_inv_filt, g=None):
    tmap = np.copy(maps)
    n_inv_filt.apply_map(tmap)
    lmax = len(n_inv_filt.tf1d) - 1
    if g is None:
        alm = map2alm(tmap, lmax=lmax, iter=0)
    else:
        alm = g.map2alm(tmap, lmax=lmax, iter=0)
    if n_inv_filt.tf2d is None:
        hp.almxfl(alm, n_inv_filt.tf1d, inplace=True)
    else:
        alm *= n_inv_filt.tf2d
    pixarea = hp.nside2pixarea(hp.get_nside(maps))
    return alm / pixarea


def calc_fini(alm, s_inv_filt, n_inv_filt):
    """This final operation turns the Wiener-filtered CMB cg-solution to the inverse-variance filtered CMB."""
    s_inv_filt.calc(alm, inplace=True)


class DotOperator:
    """Scalar product definition for cg-inversion"""

    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        lmax1 = hp.Alm.getlmax(alm1.size)
        assert lmax1 == hp.Alm.getlmax(alm2.size)
        return np.sum(hp.alm2cl(alm1, alms2=alm2) * (2.0 * np.arange(0, lmax1 + 1) + 1))


class ForwardOperator:
    """Conjugate-gradient inversion forward operation definition."""

    def __init__(self, s_inv_filt, n_inv_filt):
        self.s_inv_filt = s_inv_filt
        self.n_inv_filt = n_inv_filt

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        if np.all(talm == 0):  # do nothing if zero
            return talm
        nlm = np.copy(talm)
        self.n_inv_filt.apply_alm(nlm)
        slm = self.s_inv_filt.calc(talm)
        return nlm + slm


class PreOperatorDiag:
    """
    Harmonic space diagonal pre-conditioner operation.

    Attributes
    ----------
    filt: array_like
        1/(1/S + 1/N) used as pre-conditioner
    fl: array_like
        1/(S+N) used for QE weights
    """

    def __init__(self, s_cls, n_inv_filt, nl_res=None):
        lmax = len(n_inv_filt.tf1d) - 1
        cltt = s_cls["tt"]
        assert len(cltt) >= len(n_inv_filt.tf1d)
        assert lmax <= (len(cltt) - 1)
        if nl_res is None:
            nl_res = {key: np.zeros(lmax + 1) for key in s_cls}

        bl2 = n_inv_filt.tf1d[:lmax+1]**2
        sl = cltt[:lmax + 1]+ nl_res["tt"] * cinv_utils.cli(bl2)
        filt = cinv_utils.cli(sl)
        filt += 1/n_inv_filt.nlev_cl * bl2
        self.filt = cinv_utils.cli(filt)
        self.fl = cinv_utils.cli(sl + n_inv_filt.nlev_cl * cinv_utils.cli(bl2))
        self.fl[:2] = 0

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return hp.almxfl(talm, self.filt)


class SkyInverseFilter:  # alm_filter_sinv_nocorr:
    """Class allowing spectrum-formed covariances
    from signal and also 1/ell noise.
    For non-TT-only cases, does not include TE correlation
    """

    def __init__(self, s_cls, nl_res, lmax, tf1d, tf2d=None):
        cltt = s_cls["tt"][: lmax + 1]
        cltt_2d = hp_utils.cl2almformat(cltt)
        nltt = nl_res["tt"][: lmax + 1] / tf1d ** 2
        nltt_2d = hp_utils.cl2almformat(nltt)
        self.slinv = cinv_utils.cli(cltt_2d + nltt_2d)
        self.nl_res = nl_res
        self.lmax = lmax
        self.s_cls = s_cls
        self.tf2d = tf2d

    def calc(self, alm, inplace=False):
        # if self.n_cls is not None and self.tf2d is not None:
        if inplace:
            alm *= self.slinv
        else:
            return alm * self.slinv


class NoiseInverseFilter(opfilt_hp.NoiseInverseFilter):  # alm_filter_ninv(object):
    """Missing doc."""
    nlev_cl: float  # equivalent noise level in uK^2 sr for precond

    def __init__(self, n_inv, tf1d, tf2d=None, g=None):
        # marge_monopole=False, marge_dipole=False, marge_uptolmin=-1, marge_maps=(), nlev_ftl=None):

        ninv_std = np.std(n_inv[np.where(n_inv != 0.0)])
        ninv_avg = np.average(n_inv[np.where(n_inv != 0.0)])
        logger.info(f"inverse noise map std dev / av = {ninv_std / ninv_avg:.3e}")

        self._n_inv = n_inv
        self.tf1d = tf1d
        self.tf2d = tf2d
        self.npix = len(self.n_inv)
        self.nside = hp.npix2nside(self.npix)
        self.pixarea = hp.nside2pixarea(self.nside)

        if g is None:
            self.g = None
        else:
            self.g = g
            assert self.g.nside == self.nside

        self.nlev_cl, fsky, NET = self.ninv2nlev(self.n_inv)

    @cached_property
    def n_inv(self):
        return self.load_ninvs(self._n_inv)

    def apply_alm(self, alm):
        """Missing doc."""
        if self.tf2d is None:
            hp.almxfl(alm, self.tf1d, inplace=True)
        else:
            alm *= self.tf2d
        if self.g is None:
            tmap = alm2map(alm, self.nside)
        else:
            tmap = self.g.alm2map(alm,)
        self.apply_map(tmap)
        lmax = hp.Alm.getlmax(alm.size)
        if self.g is None:
            alm[:] = map2alm(tmap, lmax=lmax, iter=0)
        else:
            self.g.map2alm(tmap, lmax=lmax, iter=0, alms=alm)
        if self.tf2d is None:
            hp.almxfl(alm, self.tf1d, inplace=True)
        else:
            alm *= self.tf2d
        alm /= self.pixarea

    def apply_map(self, tmap):
        """Missing doc."""
        tmap *= self.n_inv
