"""
Similar to opfilt_teb.py for flatsky, this is for QU/EB filtering for healpix
maps.

Modified from and built on plancklens/qcinv/opfilt_pp.py
"""

import numpy as np
import healpy as hp

from healpy import alm2map_spin, map2alm_spin
from . import cinv_utils, hp_utils

clhash = cinv_utils.clhash
eblm = hp_utils.eblm


def calc_prep(maps, s_inv_filt, n_inv_filt):
    qmap = np.copy(hp_utils.read_map(maps[0]))
    umap = np.copy(hp_utils.read_map(maps[1]))
    assert len(qmap) == len(umap)
    lmax = len(n_inv_filt.tf1dE) - 1
    npix = len(qmap)

    n_inv_filt.apply_map([qmap, umap])
    elm, blm = map2alm_spin([qmap, umap], 2, lmax=lmax)
    if n_inv_filt.tf2dE is None or n_inv_filt.tf2dB is None:
        hp.almxfl(elm, n_inv_filt.tf1dE * npix / (4.0 * np.pi), inplace=True)
        hp.almxfl(blm, n_inv_filt.tf1dB * npix / (4.0 * np.pi), inplace=True)
    else:
        elm *= n_inv_filt.tf2dE * npix / (4.0 * np.pi)
        blm *= n_inv_filt.tf2dB * npix / (4.0 * np.pi)
    return eblm([elm, blm])


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
        clbb = s_cls["bb"][: lmax + 1]
        clee = s_cls["ee"][: lmax + 1]
        if nl_res is None:
            nl_res = {key: np.zeros(lmax + 1) for key in s_cls}

        ninv_fel, ninv_fbl = n_inv_filt.get_febl()

        filt_e = cinv_utils.cli(
            clee + nl_res["ee"] * cinv_utils.cli(n_inv_filt.tf1dE[: lmax + 1] ** 2)
        )
        filt_e += ninv_fel[: lmax + 1]
        filt_b = cinv_utils.cli(
            clbb + nl_res["bb"] * cinv_utils.cli(n_inv_filt.tf1dB[: lmax + 1] ** 2)
        )
        filt_b += ninv_fbl[: lmax + 1]

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


class NoiseInverseFilter:
    def __init__(self, n_inv, tf1dE, tf1dB, tf2dE, tf2dB, nlev_febl=None):
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

        self.tf1dE = tf1dE
        self.tf1dB = tf1dB

        self.tf2dE = tf2dE
        self.tf2dB = tf2dB

        # These three things will be instantiated later on
        self.nside = None
        self.n_inv = None

        self.nlev_febl = nlev_febl
        self._n_inv = n_inv  # could be paths or list of paths
        self._load_ninv()

    def _load_ninv(self):
        if self.n_inv is None:
            self.n_inv = []
            for i, tn in enumerate(self._n_inv):
                if isinstance(tn, list):
                    n_inv_prod = hp_utils.read_map(tn[0])
                    if len(tn) > 1:
                        for n in tn[1:]:
                            n_inv_prod = n_inv_prod * hp_utils.read_map(n)
                    self.n_inv.append(n_inv_prod)
                else:
                    self.n_inv.append(hp_utils.read_map(self._n_inv[i]))
            assert len(self.n_inv) in [1, 3], len(self.n_inv)
            self.nside = hp.npix2nside(len(self.n_inv[0]))

    def _calc_febl(self):
        self._load_ninv()
        if len(self.n_inv) == 1:
            nlev_febl = 10800.0 / np.sqrt(np.sum(self.n_inv[0]) / (4.0 * np.pi)) / np.pi
        elif len(self.n_inv) == 3:
            nlev_febl = (
                10800.0
                / np.sqrt(np.sum(0.5 * (self.n_inv[0] + self.n_inv[2])) / (4.0 * np.pi))
                / np.pi
            )
        else:
            assert 0
        print("ninv_febl: using %.2f uK-amin noise Cl" % nlev_febl)
        return nlev_febl

    def get_ninv(self):
        self._load_ninv()
        return self.n_inv

    def get_mask(self):
        ninv = self.get_ninv()
        assert len(ninv) in [1, 3], len(ninv)
        self.nside = hp.npix2nside(len(ninv[0]))
        mask = np.where(ninv[0] > 0, 1.0, 0)
        for ni in ninv[1:]:
            mask *= ni > 0
        return mask

    def get_febl(self):
        if self.nlev_febl is None:
            self.nlev_febl = self._calc_febl()
        n_inv_cl_e = self.tf1dE**2 / (self.nlev_febl / 180.0 / 60.0 * np.pi) ** 2
        n_inv_cl_b = self.tf1dB**2 / (self.nlev_febl / 180.0 / 60.0 * np.pi) ** 2
        return n_inv_cl_e, n_inv_cl_b

    def apply_alm(self, alm):
        """B^dagger N^{-1} B"""
        self._load_ninv()
        lmax = alm.lmax

        if self.tf2dE is None and self.tf2dB is None:
            hp.almxfl(alm.elm, self.tf1dE, inplace=True)
            hp.almxfl(alm.blm, self.tf1dB, inplace=True)
        else:
            alm.elm *= self.tf2dE
            alm.blm *= self.tf2dB
        qmap, umap = alm2map_spin((alm.elm, alm.blm), self.nside, 2, lmax)

        self.apply_map([qmap, umap])  # applies N^{-1}
        npix = len(qmap)

        telm, tblm = map2alm_spin([qmap, umap], 2, lmax=lmax)
        alm.elm[:] = telm
        alm.blm[:] = tblm

        if self.tf2dE is None and self.tf2dB is None:
            hp.almxfl(alm.elm, self.tf1dE * (npix / (4.0 * np.pi)), inplace=True)
            hp.almxfl(alm.blm, self.tf1dB * (npix / (4.0 * np.pi)), inplace=True)
        else:
            alm.elm *= self.tf2dE * (npix / (4.0 * np.pi))
            alm.blm *= self.tf2dB * (npix / (4.0 * np.pi))

    def apply_map(self, amap):
        self._load_ninv()
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
