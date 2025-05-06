"""
Similar to opfilt_teb.py for flatsky, this is for QU/EB filtering for healpix
maps.

Modified from and built on plancklens/qcinv/opfilt_pp.py
"""

import logging
import numpy as np
import healpy as hp
from functools import cached_property

from . import cinv_utils, hp_utils, opfilt_hp
from .cinv_utils import clhash
from .hp_utils import eblm
from .opfilt_hp import alm2map, map2alm
logger = logging.getLogger(__name__)


def calc_prep(maps, s_inv_filt, n_inv_filt, g=None):
    qmap = np.copy(hp_utils.read_map(maps[0]))
    umap = np.copy(hp_utils.read_map(maps[1]))
    assert len(qmap) == len(umap)
    lmax = len(n_inv_filt.tf1dE) - 1

    n_inv_filt.apply_map([qmap, umap])
    if g is None:
        elm, blm = map2alm([qmap, umap],  lmax=lmax)
    else:
        elm, blm = g.map2alm_spin([qmap, umap], 2, lmax=lmax)
    if n_inv_filt.tf2dE is None or n_inv_filt.tf2dB is None:
        hp.almxfl(elm, n_inv_filt.tf1dE, inplace=True)
        hp.almxfl(blm, n_inv_filt.tf1dB, inplace=True)
    else:
        elm *= n_inv_filt.tf2dE
        blm *= n_inv_filt.tf2dB
    pixarea = hp.nside2pixarea(hp.get_nside(maps))
    return eblm([elm/pixarea, blm/pixarea])


def calc_fini(alm, s_inv_filt, n_inv_filt):
    ret = s_inv_filt.calc(alm)
    alm.elm[:] = ret.elm[:]
    alm.blm[:] = ret.blm[:]


class DotOperator:
    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        assert alm1.lmax == alm2.lmax
        tcl = hp.alm2cl(alm1.elm, alm2.elm) + hp.alm2cl(alm1.blm, alm2.blm)
        return np.sum(tcl[2:] * (2.0 * np.arange(2, alm1.lmax + 1) + 1))


class ForwardOperator:
    """Missing doc."""

    def __init__(self, s_inv_filt, n_inv_filt):
        self.s_inv_filt = s_inv_filt
        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return {
            "s_inv_filt": self.s_inv_filt.hashdict(),
            "n_inv_filt": self.n_inv_filt.hashdict(),
        }

    def __call__(self, alm):
        return self.calc(alm)

    def calc(self, alm):
        nlm = alm * 1.0
        self.n_inv_filt.apply_alm(nlm)
        slm = self.s_inv_filt.calc(alm)
        return nlm + slm


class PreOperatorDiag:
    """Missing doc."""

    def __init__(self, s_cls, n_inv_filt, nl_res=None):
        lmax = len(n_inv_filt.tf1dE) - 1
        clbb = s_cls["bb"][:lmax + 1]
        clee = s_cls["ee"][:lmax + 1]
        if nl_res is None:
            nl_res = {key: np.zeros(lmax + 1) for key in s_cls}

        bl2ee = n_inv_filt.tf1dE[:lmax+1]**2
        bl2bb = n_inv_filt.tf1dB[:lmax+1]**2
        filt_e = cinv_utils.cli(clee + nl_res["ee"] * cinv_utils.cli(bl2ee))
        filt_b = cinv_utils.cli(clbb + nl_res["bb"] * cinv_utils.cli(bl2bb))

        filt_e += 1/n_inv_filt.nlev_cl * bl2ee
        filt_b += 1/n_inv_filt.nlev_cl * bl2bb

        self.filt_e = cinv_utils.cli(filt_e)
        self.filt_b = cinv_utils.cli(filt_b)

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, alm):
        relm = hp.almxfl(alm.elm, self.filt_e)
        rblm = hp.almxfl(alm.blm, self.filt_b)
        return eblm([relm, rblm])


class SkyInverseFilter:  # alm_filter_sinv_nocorr:
    def __init__(self, s_cls, nl_res, lmax, tf1dE, tf1dB, tf2dE, tf2dB):
        clee = s_cls.get("ee", np.zeros(lmax + 1))[: lmax + 1]
        clbb = s_cls.get("bb", np.zeros(lmax + 1))[: lmax + 1]

        nlee = nl_res.get("ee", np.zeros(lmax + 1))[: lmax + 1] / tf1dE**2
        nlbb = nl_res.get("bb", np.zeros(lmax + 1))[: lmax + 1] / tf1dB**2

        clee_2d = hp_utils.cl2almformat(clee)
        clbb_2d = hp_utils.cl2almformat(clbb)

        nlee_2d = hp_utils.cl2almformat(nlee)
        nlbb_2d = hp_utils.cl2almformat(nlbb)

        self.e_slinv = cinv_utils.cli(clee_2d + nlee_2d)
        self.b_slinv = cinv_utils.cli(clbb_2d + nlbb_2d)

        self.lmax = lmax
        self.s_cls = s_cls
        self.tf2dE = tf2dE
        self.tf2dB = tf2dB

        self.nl_res = nl_res

    def calc(self, alm):
        relm = alm.elm * self.e_slinv
        rblm = alm.blm * self.b_slinv

        return eblm([relm, rblm])


class NoiseInverseFilter(opfilt_hp.NoiseInverseFilter):
    nlev_cl: float  # equivalent noise level in uK^2 sr for precond

    def __init__(self, n_inv, tf1dE, tf1dB, tf2dE, tf2dB, g=None):
        # , marge_qmaps=(), marge_umaps=()):
        """Inverse-variance filtering instance for polarization only

        Args:
            n_inv: inverse pixel variance maps or masks
            b_transf: filter fiducial transfer function
            nlev_febl(optional): isotropic approximation to the noise level across the entire map
                                 this is used e.g. in the diag. preconditioner of cg inversion.
            b_transf_b: B-mode transfer func if different from E-mode

        Note:
            This allows for independent Q and U map marginalization

        """

        # self.nlev_febl = nlev_febl
        self._n_inv = n_inv  # could be paths or list of paths
        self.tf1dE = tf1dE
        self.tf1dB = tf1dB
        self.tf2dE = tf2dE
        self.tf2dB = tf2dB
        self.npix = len(self.n_inv[0])
        self.nside = hp.npix2nside(self.npix)
        self.pixarea = hp.nside2pixarea(self.nside)

        if g is None:
            self.g = None
        else:
            self.g = g
            assert self.g.nside == self.nside

        ninv = (self.n_inv[0] + self.n_inv[-1]) / 2  # QU avg ninv
        self.nlev_cl, fsky, NET = self.ninv2nlev(ninv)

    @cached_property
    def n_inv(self):
        out = []
        for i, tn in enumerate(self._n_inv):
            out.append(self.load_ninvs(tn))
        assert len(out) in [1, 3], len(out)
        return out

    def apply_alm(self, alm):
        """B^dagger N^{-1} B"""
        lmax = alm.lmax

        if self.tf2dE is None and self.tf2dB is None:
            hp.almxfl(alm.elm, self.tf1dE, inplace=True)
            hp.almxfl(alm.blm, self.tf1dB, inplace=True)
        else:
            alm.elm *= self.tf2dE
            alm.blm *= self.tf2dB

        if self.g is None:
            qmap, umap = alm2map((alm.elm, alm.blm), self.nside, lmax)
        else:
            qmap, umap = self.g.alm2map_spin((alm.elm, alm.blm), 2, lmax)

        self.apply_map([qmap, umap])  # applies N^{-1}

        if self.g is None:
            telm, tblm = map2alm([qmap, umap], lmax=lmax)
        else:
            telm, tblm = self.g.map2alm_spin([qmap, umap], 2, lmax=lmax)
        alm.elm[:] = telm
        alm.blm[:] = tblm

        if self.tf2dE is None and self.tf2dB is None:
            hp.almxfl(alm.elm, self.tf1dE / self.pixarea, inplace=True)
            hp.almxfl(alm.blm, self.tf1dB / self.pixarea, inplace=True)
        else:
            alm.elm *= self.tf2dE / self.pixarea
            alm.blm *= self.tf2dB / self.pixarea

    def apply_map(self, amap):
        [qmap, umap] = amap
        if len(self.n_inv) == 1:  # TT, QQ=UU
            qmap *= self.n_inv[0]
            umap *= self.n_inv[0]

        elif len(self.n_inv) == 3:  # TT, QQ, QU, UU
            qmap_copy = qmap.copy()

            qmap *= self.n_inv[0]
            qmap += self.n_inv[1] * umap

            umap *= self.n_inv[2]
            umap += self.n_inv[1] * qmap_copy

            del qmap_copy
        else:
            assert 0
