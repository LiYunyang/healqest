"""
Modified from and built on plancklens/qcinv/opfilt_tt.py

Mergerd from opfilt_hp_[t/p/tp].py
"""

import logging
import numpy as np
from numpy.typing import NDArray
import healpy as hp
from healqest import ducc_sht
from . import hp_utils, cinv_utils
logger = logging.getLogger(__name__)


# TODO: for testing only
def alm2map(alms, *args, **kwargs):
    logger.warning("using healpy alm2map")
    if alms.ndim == 2 and alms.shape[0] == 2:
        lmax = hp.Alm.getlmax(alms.shape[-1])
        return np.array(hp.alm2map_spin(alms, *args, **kwargs, spin=2, lmax=lmax))
    else:
        assert alms.ndim == 1 or alms.shape[0] in [1, 3], f"{alms.shape}"
        return hp.alm2map(alms, *args, **kwargs)


# TODO: for testing only
def map2alm(maps, *args, **kwargs):
    logger.warning("using healpy map2alm")
    if maps.ndim == 2 and maps.shape[0] == 2:
        if 'iter' in kwargs:
            kwargs.pop('iter')
        return np.array(hp.map2alm_spin(maps, *args, **kwargs, spin=2))
    else:
        assert maps.ndim == 1 or maps.shape[0] in [1, 3], f"{maps.shape}"
        return hp.map2alm(maps, *args, **kwargs)


def calc_fini(alms, s_inv_filt):
    """This final operation turns the Wiener-filtered CMB cg-solution to the inverse-variance filtered CMB."""
    s_inv_filt.calc(alms, inplace=True)


class TFObj:
    """Transfer function object, takes list of tf1d and tf2d and parse them into T/E/B TFs"""
    lmax: int
    tf1d_t: NDArray = None  # (lmax+1, )
    tf1d_e: NDArray = None  # (lmax+1, )
    tf1d_b: NDArray = None  # (lmax+1, )
    tf2d_t: NDArray = None  # (almsize, )
    tf2d_e: NDArray = None  # (almsize, )
    tf2d_b: NDArray = None  # (almsize, )
    bl_t: NDArray = None  # (lmax+1, )
    bl_e: NDArray = None  # (lmax+1, )
    bl_b: NDArray = None  # (lmax+1, )
    lx_cut = None
    m_cut = None

    def __init__(self, npol, lmax, tf1d, tf2d=None, bl=None, lx_cut=None, m_cut=None):
        """
        Parameters
        ----------
        npol : int
            Number of polarizations. 1 for T, 2 for E/B, 3 for T/E/B
        tf1d: array or list of array
        tf2d: array or list of array
        """
        self.npol = npol
        self.lmax = lmax
        _tf1d = np.atleast_2d(tf1d)[:, :lmax+1]
        assert _tf1d.shape[-1] == lmax+1
        assert _tf1d.shape[0] in [1, 2]

        if tf2d is not None and tf2d != [None, None]:
            _tf2d = np.atleast_2d(tf2d)
            assert _tf2d.shape[0] in [1, 2]
            assert hp.Alm.getlmax(_tf2d.shape[-1]) == lmax
        else:
            _tf2d = np.array([None, None])

        if bl is not None:
            _bl = np.atleast_2d(bl)[:, :lmax+1]
            assert _bl.shape[0] in [1, 2]
        else:
            _bl = [None, None]

        # mmcut should've already been incoporated in tf2d in config. So no just need to apply m_cut to mtheta.
        if lx_cut is not None:
            assert bl is not None, "lx_cut requires bl to be set"
            self.lx_cut = lx_cut
            self.m_cut = m_cut
            if tf2d is not None:
                logger.warning(f"using lx_cut={lx_cut} in lieu of the provided tf2d")
                _tf2d = np.array([None, None])
                tf2d = None

        if npol == 1:
            # t-only case
            self.tf1d_t = _tf1d[0]
            self.tf2d_t = _tf2d[0]
            self.bl_t = _bl[0]
            logger.debug(f"raw tf1d shape: {_tf1d.shape}, taking first row for T")
            if tf2d is not None:
                logger.debug(f"raw tf2d shape: {_tf2d.shape}, taking first row for T")
        elif npol == 2:
            # pol-only case
            self.tf1d_e, self.tf1d_b = _tf1d[0], _tf1d[-1]
            self.tf2d_e, self.tf2d_b = _tf2d[0], _tf2d[-1]
            self.bl_e, self.bl_b = _bl[0], _bl[-1]
            logger.debug(f"raw tf1d shape: {_tf1d.shape}, taking first/last row for E/B")
            if tf2d is not None:
                logger.debug(f"raw tf2d shape: {_tf2d.shape}, taking first/last row for E/B")
        elif npol == 3:
            self.tf1d_t, self.tf1d_e, self.tf1d_b = _tf1d[0], _tf1d[-1], _tf1d[-1]
            self.tf2d_t, self.tf2d_e, self.tf2d_b = _tf2d[0], _tf2d[-1], _tf2d[-1]
            self.bl_t, self.bl_e, self.bl_b = _bl[0], _bl[-1], _bl[-1]
            logger.debug(f"raw tf1d shape: {_tf1d.shape}, taking first row for T, last row for EB")
            if tf2d is not None:
                logger.debug(f"raw tf2d shape: {_tf2d.shape}, taking first row for T, last row for EB")
        else:
            raise ValueError(f"npol must be 1/2/3, not {npol}")

    @property
    def pols(self):
        if self.npol == 1:
            return ['t']
        elif self.npol == 2:
            return ['e', 'b']
        elif self.npol == 3:
            return ['t', 'e', 'b']

    def apply_tf(self, alms, maps, g, adjoint=False, g_tf=None):
        """
        Apply the 2d/1d (if 2d is not set) TF to alms (forward operation) or maps (adjoint operation)

        alms: array_like, optional
            Needed for forward operation, but can be omitted for adjoint operation. If provided in the latter case,
            it will be used as a buffer.
        maps: array_like
            Needed for adjoint operation, but can be omitted for forward operation. If provided in the latter case,
            it will be used as a buffer.
        g: geometry object
        """
        if adjoint:
            assert isinstance(maps, np.ndarray)
        else:
            assert isinstance(alms, np.ndarray)
        # atleast_2d creates a view (for inplace modification), only if the input is not already numpy array!

        # === start of adjoint operation, but not needed forward operation ===
        if adjoint:
            if self.lx_cut:
                # slice assignment is important to make sure that it does modification inplace!
                maps[:] = g_tf.apply_map(np.atleast_2d(maps))
            alms = g.map2alm(maps, lmax=self.lmax, iter=0, alms=alms, check=False)

        # === alm space operation, applying tf2d/tf1d or just bl. ===
        alms = np.atleast_2d(alms)
        for i, s in enumerate(self.pols):
            tf2d = getattr(self, f"tf2d_{s}")
            if tf2d is None:
                if self.lx_cut:
                    # perform beam operation if lx_cut will be applied
                    tf1d = getattr(self, f"bl_{s}")
                else:
                    # otherwise, perform with the effective 1d TF (that includes bl!)
                    tf1d = getattr(self, f"tf1d_{s}")
                hp.almxfl(alms[i], tf1d, inplace=True)
            else:
                alms[i] *= tf2d

        # === end of adjoint operation, but do convert to maps for forward operation ===
        if adjoint:
            return np.squeeze(alms)
        else:
            maps = g.alm2map(alms, maps=maps)
            if self.lx_cut:
                maps[:] = g_tf.apply_map(np.atleast_2d(maps))
            return np.squeeze(maps)

    def __imul__(self, fl):
        """scale transfer functions by a common array"""
        for s in self.pols:
            # setattr modifies each copy of the tf1d/tf2d/bl SEPARATELY.
            # so no worries that each attribute begins as a view of the same array.
            tf1d = getattr(self, f"tf1d_{s}")
            tf2d = getattr(self, f"tf2d_{s}")
            bl = getattr(self, f"bl_{s}")
            setattr(self, f"tf1d_{s}", tf1d*fl)
            if tf2d is not None:
                setattr(self, f"tf2d_{s}", hp.almxfl(tf2d, fl))
            if bl is not None:
                setattr(self, f"bl_{s}", bl*fl)
        return self


class DotOperator:
    """Scalar product definition for cg-inversion"""

    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        alm1 = np.atleast_2d(alm1)
        alm2 = np.atleast_2d(alm2)
        lmax = hp.Alm.getlmax(alm1.shape[-1])
        assert alm1.shape == alm2.shape
        tcl = np.array([hp.alm2cl(alm1[i], alm2[i]) for i in range(alm1.shape[0])])
        ell = np.arange(lmax+1)
        return np.einsum('ij,j->', tcl, (2.0 * ell + 1))


class ForwardOperator:
    """Conjugate-gradient inversion forward operation definition."""

    def __init__(self, s_inv_filt, n_inv_filt):
        self.s_inv_filt = s_inv_filt
        self.n_inv_filt = n_inv_filt

    def __call__(self, alms):
        return self.calc(alms)

    def calc(self, alms):
        """Not altering alms!"""
        nlm = np.copy(alms)
        self.n_inv_filt.apply_alm(nlm)
        slm = self.s_inv_filt.calc(alms)
        return nlm + slm


class PreOperatorDiag:
    """
    Harmonic space diagonal pre-conditioner operation

    Attributes
    ----------
    filt: array_like
        1/(1/S + bl2/N) used as pre-conditioner
    fl: array_like
        1/(S+N/bl2) used for QE weights
    """
    def __init__(self, s_cls, n_inv_filt, tf, nl_res=None):

        self.tf = tf
        self.cls = dict()
        lmax = self.tf.lmax

        if nl_res is None:
            nl_res = {key: np.zeros(lmax + 1) for key in s_cls}

        filt = list()
        fl = list()
        for i, s in enumerate(self.tf.pols):
            ss = f"{s}{s}"
            cl = s_cls[ss][:lmax + 1]
            nl = nl_res[ss][:lmax + 1]

            bl2 = getattr(tf, f"tf1d_{s}")[:lmax+1]**2
            sl = cl+ nl / bl2
            self.cls[s] = sl
            nlev = n_inv_filt.nlev_cl_t if s == 't' else n_inv_filt.nlev_cl_p
            _filt = 1/sl + 1/nlev * bl2

            _fl = cinv_utils.cli(sl + nlev / bl2)  # direct bl2 division is better than NESTED cli!
            _fl[:2] = 0
            filt.append(_filt)
            fl.append(_fl)

        self.filt = cinv_utils.cli(np.array(filt))
        self.fl = np.array(fl)

    def __call__(self, alms):
        return self.calc(alms)

    def calc(self, alms):
        alms = np.atleast_2d(alms)
        assert alms.shape[0] == self.filt.shape[0]
        almsf = [hp.almxfl(alms[i], self.filt[i]) for i in range(alms.shape[0])]
        return np.squeeze(almsf)

    def get_initial_guess(self, maps, rescale_cl, g=None):
        """
        The initial guess is S/bl/(N/bl2+S)/rescale_cl
        """
        # HACK: temporary
        logger.warning("making educated guess as the initial solution")

        alms = np.atleast_2d(g.map2alm(maps, lmax=self.tf.lmax))
        assert alms.shape[0] == len(self.tf.pols)
        for idx, s in enumerate(self.tf.pols):
            fl = self.cls[s]/rescale_cl*self.fl[idx]
            fl *= cinv_utils.cli(getattr(self.tf, f"tf1d_{s}"))
            hp.almxfl(alms[idx], fl, inplace=True)
        return np.squeeze(alms)


class PreOperatorDiagJoint(PreOperatorDiag):
    """
    Harmonic space diagonal pre-conditioner operation for TEB with TE correlation

    Attributes
    ----------
    filt: array_like
        (S^{-1} + bl2/N)^-1 used as pre-conditioner
    fl: array_like
        1/(S+N/bl2) used for QE weights
    """
    def __init__(self, s_cls, n_inv_filt, tf, nl_res=None, te_only=True):
        if not te_only:
            raise NotImplementedError("Only TE-only case is implemented")

        self.tf = tf
        assert self.tf.npol == 3, "Joint preconditioner needs all T/E/B"

        self.cls = dict()
        lmax = self.tf.lmax

        if nl_res is None:
            nl_res = {key: np.zeros(lmax + 1) for key in s_cls}

        fl = list()
        s_mat = np.zeros((3, lmax + 1))
        n_mat = np.zeros((3, lmax + 1))
        for i, s in enumerate('teb'):
            ss = f"{s}{s}"
            cl = s_cls[ss][:lmax + 1]
            nl = nl_res[ss][:lmax + 1]
            bl2 = getattr(tf, f"tf1d_{s}")[:lmax + 1] ** 2
            s_mat[i] = cl + nl / bl2

            nlev = n_inv_filt.nlev_cl_t if s == 't' else n_inv_filt.nlev_cl_p
            n_mat[i] = 1 / nlev * bl2

            _fl = cinv_utils.cli(s_mat[i] + nlev / bl2)
            _fl[:2] = 0
            fl.append(_fl)

        blte2 = (getattr(tf, f"tf1d_t") * getattr(tf, f"tf1d_e"))[:lmax + 1]
        _te = s_cls['te'][:lmax + 1] + nl_res['te'][:lmax + 1] / blte2
        sinv, sinv_te = cinv_utils.invert_teb(np.array(s_mat), te=_te)
        self.filt, self.filt_te = cinv_utils.invert_teb(sinv + n_mat, te=sinv_te)
        self.fl = np.array(fl)

        # for TE part (only used for GMV resp)
        cl = s_cls['te'][:lmax + 1]
        nl = nl_res['te'][:lmax + 1]
        bl2 = tf.tf1d_t[:lmax + 1]*tf.tf1d_e[:lmax + 1]
        self.fl_te = cinv_utils.cli(cl + nl / bl2)
        self.fl_te[:2] = 0

    def calc(self, alms):
        alms = np.atleast_2d(alms)
        assert alms.shape[0] == self.filt.shape[0]
        almsf = [hp.almxfl(alms[i], self.filt[i]) for i in range(alms.shape[0])]
        almsf[0] += hp.almxfl(alms[1], self.filt_te)
        almsf[1] += hp.almxfl(alms[0], self.filt_te)
        return np.squeeze(almsf)


class SkyInverseFilter:  # alm_filter_sinv_nocorr:
    """ class that performs single (+1/ell noise) inverse filtering: S^-1 for T/EB or TEB without TE correlation"""

    def __init__(self, s_cls, nl_res, tf):
        self.tf = tf
        self.lmax = tf.lmax
        self.s_cls = s_cls
        self.nl_res = nl_res

        slinv = list()
        for i, s in enumerate(self.tf.pols):
            ss = f"{s}{s}"
            cltt = s_cls[ss][:self.lmax + 1]
            cltt_2d = hp_utils.cl2almformat(cltt)
            tf1d = getattr(self.tf, f"tf1d_{s}")
            nltt = nl_res[ss][:self.lmax + 1]/tf1d**2
            nltt_2d = hp_utils.cl2almformat(nltt)
            slinv.append(cinv_utils.cli(cltt_2d + nltt_2d))
        self.slinv = np.array(slinv)

    def calc(self, alms, inplace=False):
        assert isinstance(alms, np.ndarray)
        # atleast_2d creates a view (for inplace modification), only if
        # the input is not already numpy array!
        alms = np.atleast_2d(alms)
        if inplace:
            alms *= self.slinv
        else:
            return np.squeeze(alms * self.slinv)


class SkyInverseFilterJoint(SkyInverseFilter):
    """ class that performs single (+1/ell noise) inverse filtering: S^-1 for TEB with TE correlation"""
    def __init__(self, s_cls, nl_res, tf):
        self.tf = tf
        self.lmax = tf.lmax
        self.s_cls = s_cls
        self.nl_res = nl_res
        almsize = hp.Alm.getsize(self.lmax)

        slinv = np.zeros((3, almsize), dtype=float)
        for i, s in enumerate('teb'):
            ss = f"{s}{s}"
            cl = s_cls[ss][:self.lmax + 1]
            nl = nl_res[ss][:self.lmax + 1]
            bl2 = getattr(tf, f"tf1d_{s}")[:self.lmax + 1] ** 2
            slinv[i] = hp_utils.cl2almformat(cl + nl / bl2)

        blte2 = (getattr(tf, f"tf1d_t") * getattr(tf, f"tf1d_e"))[:self.lmax + 1]
        _te = s_cls['te'][:self.lmax + 1] + nl_res['te'][:self.lmax + 1] / blte2
        _te = hp_utils.cl2almformat(_te)
        self.slinv, self.slinv_te = cinv_utils.invert_teb(slinv, te=_te)

    def calc(self, alms, inplace=False):
        # This is the case where EB=BE=TB=BT==0
        alms_out = alms*self.slinv
        alms_out[0] += alms[1]*self.slinv_te
        alms_out[1] += alms[0] * self.slinv_te
        if inplace:
            alms[:] = alms_out
            return
        return alms_out


class NoiseInverseFilter:  # alm_filter_ninv(object):
    """class that performs inverse variance filtering: [tfbl] [m2a] N-1 [a2m] [tfbl]"""
    nlev_cl_t: float = None  # equivalent noise level in uK^2 sr for precond
    nlev_cl_p: float = None  # equivalent noise level in uK^2 sr for precond
    almsize: int

    def __init__(self, n_inv, tf, g=None, fast=True, **kwargs):
        """
        Parameters
        ----------
        n_inv : array_like
            Inverse noise map, (npix, ) or (2, npix). In the latter case, the two rows represent NET and NEQ/U
        tf: TFObj
        g: Geometry
            ducc wrapper to speed up alm2map/map2alm
        fast: bool=False
            If True, only store partial maps. This should significantly speedup `apply_map`
        """

        _ninv = np.atleast_2d(n_inv)
        assert _ninv.shape[0] in [1, 2, 3]
        self.nonzero = np.where(np.sum(_ninv, axis=0)>0)[0]
        # saving index is fastest compared to masks for nside2048 masks

        self.tf = tf
        self.lmax = tf.lmax
        self.almsize = hp.Alm.getsize(self.lmax)
        self.npix = _ninv.shape[-1]
        self.nside = hp.npix2nside(self.npix)
        self.pixarea = hp.nside2pixarea(self.nside)

        fsky = np.count_nonzero(self.nonzero)/_ninv.shape[-1]
        if fast:
            _ninv = np.ascontiguousarray(_ninv[:, self.nonzero])
        self.n_inv_t = None
        self.n_inv_q = None
        self.n_inv_u = None

        self.npol = tf.npol
        assert self.npol in [1, 2, 3]
        assert self.npol == _ninv.shape[0], f"ninv maps should match npol={self.npol}, got {_ninv.shape[0]}"
        if self.npol in [1, 3]:
            logger.warning(f"input ninv has shape {_ninv.shape}, taking the first row as T")
            self.n_inv_t = _ninv[0]
            self.nlev_cl_t, fsky, NET = self.ninv2nlev(self.n_inv_t, fsky=fsky)
        if self.npol in [2, 3]:
            logger.warning(f"input ninv has shape {_ninv.shape}, taking the second to last as Q")
            self.n_inv_q = _ninv[-2]
            logger.warning(f"input ninv has shape {_ninv.shape}, taking the last row as U")
            self.n_inv_u = _ninv[-1]
            self.nlev_cl_p, fsky, NET = self.ninv2nlev((self.n_inv_q+self.n_inv_u)/2, fsky=fsky)

        if g is None:
            self.g = None
        else:
            self.g = g
            assert self.g.nside == self.nside

        # setup the TF object for the theta-dependent m-cut
        if tf.lx_cut:
            self.g_tf = ducc_sht.GeometryTF(self.g, self.nonzero, lx_cut=tf.lx_cut, m_cut=tf.m_cut)
        else:
            self.g_tf = None

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
    def ninv2nlev(ninv, fsky=None):
        if fsky is None:
            fsky = np.mean(ninv > 0)
        else:
            # YL: this option is left for debugging
            pass
        nlev = 1 / np.sum(ninv) * 4 * np.pi * fsky
        NET = np.rad2deg(np.sqrt(nlev)) * 60
        logger.info(f"ninv2nlev: {NET:.2f} uK-amin noise Cl over fsky {fsky:.2f}")
        return nlev, fsky, NET

    def calc_prep(self, maps):
        maps_copy = np.copy(maps)
        self.apply_map(maps_copy)
        alms = self.tf.apply_tf(maps=maps_copy, alms=None, g=self.g, adjoint=True, g_tf=self.g_tf)
        return alms

    def apply_map(self, maps):
        """map-based Ninv operation: N^-1"""
        if maps.ndim == 1:
            maps[self.nonzero] *= self.n_inv_t/self.pixarea
        else:
            if maps.shape[0] in (1, 3):
                maps[0, self.nonzero] *= self.n_inv_t/self.pixarea
            if maps.shape[0] in (2, 3):
                maps[-2, self.nonzero] *= self.n_inv_q/self.pixarea
                maps[-1, self.nonzero] *= self.n_inv_u/self.pixarea

    def apply_alm(self, alms):
        """harmonic-space Ninv operation: apply A^T N^-1 A on alms"""
        assert alms.shape[-1] == self.almsize
        maps = self.tf.apply_tf(alms=alms, maps=None, g=self.g, adjoint=False, g_tf=self.g_tf)
        self.apply_map(maps)
        alms[:] = self.tf.apply_tf(maps=maps, alms=alms, g=self.g, adjoint=True, g_tf=self.g_tf)
