"""
Similar to opfilt_teb.py for flatsky, this is for T-only and for healpix maps.

Modified from and built on plancklens/qcinv/opfilt_tt.py
"""

import logging
import numpy as np
import healpy as hp

from . import hp_utils, cinv_utils
logger = logging.getLogger(__name__)


# TODO: for testing only
def alm2map(*args, **kwargs):
    logger.warning("using healpy alm2map")
    return hp.alm2map(*args, **kwargs)


# TODO: for testing only
def map2alm(*args, **kwargs):
    logger.warning("using healpy map2alm")
    return hp.map2alm(*args, **kwargs)


def calc_prep(m, s_inv_filt, n_inv_filt, g=None):
    tmap = np.copy(m)
    n_inv_filt.apply_map(tmap)
    if g is None:
        alm = map2alm(tmap, lmax=len(n_inv_filt.tf1d) - 1, iter=0)
    else:
        alm = g.map2alm(tmap, lmax=len(n_inv_filt.tf1d) - 1, iter=0)
    if n_inv_filt.tf2d is None:
        hp.almxfl(alm, n_inv_filt.tf1d, inplace=True)
    else:
        alm *= n_inv_filt.tf2d
    pixarea = hp.nside2pixarea(hp.get_nside(m))
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
    def __init__(self, s_cls, n_inv_filt, nl_res=None):
        """Harmonic space diagonal pre-conditioner operation."""
        """returns  1/(1/S + 1/N)"""
        cltt = s_cls["tt"]
        assert len(cltt) >= len(n_inv_filt.tf1d)
        # fsky = np.mean(n_inv_filt.n_inv>0)
        # logger.debug(f"fsky = {fsky}")
        fsky = 1
        n_inv_cl = np.sum(n_inv_filt.n_inv) / (4.0 * np.pi*fsky)
        lmax = len(n_inv_filt.tf1d) - 1
        assert lmax <= (len(cltt) - 1)
        if nl_res is None:
            nl_res = {key: np.zeros(lmax + 1) for key in s_cls}

        bl2 = n_inv_filt.tf1d[:lmax+1]**2
        filt = cinv_utils.cli(cltt[:lmax + 1]+ nl_res["tt"] * cinv_utils.cli(bl2))
        filt += n_inv_cl*bl2
        self.filt = cinv_utils.cli(filt)
        # self.filt = cltt[:lmax + 1]  # HACK (YL): simple precond

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


class NoiseInverseFilter:  # alm_filter_ninv(object):
    """Missing doc."""

    def __init__(self, n_inv, tf1d, tf2d=None, nlev_ftl=None, g=None):
        # marge_monopole=False, marge_dipole=False, marge_uptolmin=-1, marge_maps=(), nlev_ftl=None):
        if isinstance(n_inv, list):
            n_inv_prod = hp_utils.read_map(n_inv[0])
            if len(n_inv) > 1:
                for n in n_inv[1:]:
                    n_inv_prod = n_inv_prod * hp_utils.read_map(n)
            n_inv = n_inv_prod
        else:
            n_inv = hp_utils.read_map(n_inv)

        ninv_std = np.std(n_inv[np.where(n_inv != 0.0)])
        ninv_avg = np.average(n_inv[np.where(n_inv != 0.0)])
        logger.info(f"inverse noise map std dev / av = {ninv_std / ninv_avg:.3e}")

        self.n_inv = n_inv
        self.tf1d = tf1d
        self.tf2d = tf2d
        self.npix = len(self.n_inv)
        self.nside = hp.npix2nside(self.npix)
        if g is None:
            self.g = None
        else:
            self.g = g
            assert self.g.nside == self.nside
        self.pixarea = hp.nside2pixarea(self.nside)
        if nlev_ftl is None:
            nlev_ftl = np.rad2deg(np.sqrt(1 / np.sum(self.n_inv) * 4 * np.pi)) * 60
        # self.nlev_ftl = nlev_ftl
        logger.info("ninv_ftl: using %.2f uK-amin noise Cl" % nlev_ftl)

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
