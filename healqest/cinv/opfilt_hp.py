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
def alm2map(alms, *args, **kwargs):
    logger.warning("using healpy alm2map")
    if len(alms) == 2:
        return hp.alm2map_spin(alms, *args, **kwargs, spin=2)
    else:
        return hp.alm2map(alms, *args, **kwargs)


# TODO: for testing only
def map2alm(maps, *args, **kwargs):
    logger.warning("using healpy map2alm")
    if len(maps) == 2:
        return hp.map2alm_spin(maps, *args, **kwargs, spin=2)
    return hp.map2alm(maps, *args, **kwargs)


def calc_prep(maps, s_inv_filt, n_inv_filt, g=None):
    maps_copy = np.copy(maps)
    n_inv_filt.apply_map(maps_copy)
    lmax = len(n_inv_filt.tf1d) - 1
    if g is None:
        alm = map2alm(maps_copy, lmax=lmax, iter=0)
    else:
        alm = g.map2alm(maps_copy, lmax=lmax, iter=0)
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
    def __init__(self, s_cls, n_inv_filt, nl_res=None):
        """Harmonic space diagonal pre-conditioner operation."""
        """returns  1/(1/S + 1/N)"""
        cltt = s_cls["tt"]
        assert len(cltt) >= len(n_inv_filt.tf1d)

        lmax = len(n_inv_filt.tf1d) - 1
        assert lmax <= (len(cltt) - 1)
        if nl_res is None:
            nl_res = {key: np.zeros(lmax + 1) for key in s_cls}

        bl2 = n_inv_filt.tf1d[:lmax+1]**2
        filt = cinv_utils.cli(cltt[:lmax + 1]+ nl_res["tt"] * cinv_utils.cli(bl2))
        filt += 1/n_inv_filt.nlev_cl * bl2
        self.filt = cinv_utils.cli(filt)

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
    nlev_cl: float  # equivalent noise level in uK^2 sr for precond

    def __init__(self, n_inv, **kwargs):
        pass

    @staticmethod
    def load_ninvs(fnames):
        if isinstance(fnames, list):
            n_inv_prod = hp_utils.read_map(fnames[0])
            if len(fnames) > 1:
                for n in fnames[1:]:
                    n_inv_prod = n_inv_prod * hp_utils.read_map(n)
            return n_inv_prod
        else:
            return hp_utils.read_map(fnames)

    @staticmethod
    def ninv2nlev(ninv):
        fsky = np.mean(ninv > 0)
        nlev = 1 / np.sum(ninv) * 4 * np.pi * fsky
        NET = np.rad2deg(np.sqrt(nlev)) * 60
        logger.info(f"ninv2nlev: {NET:.2f} uK-amin noise Cl over fsky {fsky:.2f}")
        return nlev, fsky, NET
