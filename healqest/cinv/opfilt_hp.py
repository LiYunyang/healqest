"""
Modified from and built on plancklens/qcinv/opfilt_tt.py

Mergerd from opfilt_hp_[t/p/tp].py
"""

import logging
import numpy as np
from numpy.typing import NDArray
import healpy as hp

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
    return s_inv_filt.calc(alms, inplace=False)


class TFObj:
    """Transfer function object, takes list of tf1d and tf2d and parse them into T/E/B TFs"""
    lmax: int
    tf1d_t: NDArray = None  # (lmax+1, )
    tf1d_e: NDArray = None  # (lmax+1, )
    tf1d_b: NDArray = None  # (lmax+1, )
    tf2d_t: NDArray = None  # (almsize, )
    tf2d_e: NDArray = None  # (almsize, )
    tf2d_b: NDArray = None  # (almsize, )

    def __init__(self, npol, lmax, tf1d, tf2d=None):
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

        if tf2d is not None:
            _tf2d = np.atleast_2d(tf2d)
            assert _tf2d.shape[0] in [1, 2]
            assert hp.Alm.getlmax(_tf2d.shape[-1]) == lmax
        else:
            _tf2d = [None, None]

        if npol == 1:
            # t-only case
            self.tf1d_t = _tf1d[0]
            self.tf2d_t = _tf2d[0]
            logger.debug(f"raw tf1d shape: {_tf1d.shape}, taking first row for T")
            if tf2d is not None:
                logger.debug(f"raw tf2d shape: {_tf2d.shape}, taking first row for T")
        elif npol == 2:
            # pol-only case
            self.tf1d_e, self.tf1d_b = _tf1d[0], _tf1d[-1]
            self.tf2d_e, self.tf2d_b = _tf2d[0], _tf2d[-1]
            logger.debug(f"raw tf1d shape: {_tf1d.shape}, taking first/last row for E/B")
            if tf2d is not None:
                logger.debug(f"raw tf2d shape: {_tf2d.shape}, taking first/last row for E/B")
        elif npol == 3:
            self.tf1d_t, self.tf1d_e, self.tf1d_b = _tf1d[0], _tf1d[-1], _tf1d[-1]
            self.tf2d_t, self.tf2d_e, self.tf2d_b = _tf2d[0], _tf2d[-1], _tf2d[-1]
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
        if self.npol == 3:
            return ['t', 'e', 'b']

    def apply_tf(self, alms):
        """Apply the 2d/1d (if 2d is not set) TF to alm(s) inplace"""
        assert isinstance(alms, np.ndarray)
        # atleast_2d creates a view (for inplace modification), only if
        # the input is not already numpy array!
        alms = np.atleast_2d(alms)
        for i, s in enumerate(self.pols):
            tf1d = getattr(self, f"tf1d_{s}")
            tf2d = getattr(self, f"tf2d_{s}")
            if tf2d is None:
                hp.almxfl(alms[i], tf1d, inplace=True)
            else:
                alms[i] *= tf2d

    def __imul__(self, fl):
        """scale transfer functions by a common array"""
        for s in self.pols:
            tf1d = getattr(self, f"tf1d_{s}")
            tf2d = getattr(self, f"tf2d_{s}")
            setattr(self, f"tf1d_{s}", tf1d*fl)
            if tf2d is not None:
                setattr(self, f"tf2d_{s}", hp.almxfl(tf2d, fl))
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
            sl = cl+ nl * cinv_utils.cli(bl2)
            self.cls[s] = sl
            nlev = n_inv_filt.nlev_cl_t if s == 't' else n_inv_filt.nlev_cl_p
            _filt = cinv_utils.cli(sl) + 1/nlev * bl2

            _fl = cinv_utils.cli(sl + nlev * cinv_utils.cli(bl2))
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
            s_mat[i] = cl + nl * cinv_utils.cli(bl2)

            nlev = n_inv_filt.nlev_cl_t if s == 't' else n_inv_filt.nlev_cl_p
            n_mat[i] = 1 / nlev * bl2

            _fl = cinv_utils.cli(s_mat[i] + nlev * cinv_utils.cli(bl2))
            _fl[:2] = 0
            fl.append(_fl)

        blte2 = (getattr(tf, f"tf1d_t") * getattr(tf, f"tf1d_e"))[:lmax + 1]
        _te = s_cls['te'][:lmax + 1] + nl_res['te'][:lmax + 1] * cinv_utils.cli(blte2)
        sinv, sinv_te = cinv_utils.invert_teb(np.array(s_mat), te=_te)
        self.filt, self.filt_te = cinv_utils.invert_teb(sinv + n_mat, te=sinv_te)
        self.fl = np.array(fl)

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
            nltt = nl_res[ss][:self.lmax + 1]*cinv_utils.cli(tf1d)**2
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
            slinv[i] = hp_utils.cl2almformat(cl + nl * cinv_utils.cli(bl2))

        blte2 = (getattr(tf, f"tf1d_t") * getattr(tf, f"tf1d_e"))[:self.lmax + 1]
        _te = s_cls['te'][:self.lmax + 1] + nl_res['te'][:self.lmax + 1] * cinv_utils.cli(blte2)
        _te = hp_utils.cl2almformat(_te)
        self.slinv, self.slinv_te = cinv_utils.invert_teb(slinv, te=_te)

    def calc(self, alms, inplace=False):
        # This is the case where EB=BE=TB=BT==0
        assert not inplace
        alms_out = alms*self.slinv
        alms_out[0] += alms[1]*self.slinv_te
        alms_out[1] += alms[0] * self.slinv_te
        return alms_out


class NoiseInverseFilter:  # alm_filter_ninv(object):
    """class that performs inverse variance filtering: [tfbl] [m2a] N-1 [a2m] [tfbl]"""
    nlev_cl_t: float = None  # equivalent noise level in uK^2 sr for precond
    nlev_cl_p: float = None  # equivalent noise level in uK^2 sr for precond
    almsize: int

    def __init__(self, n_inv, tf, g=None, **kwargs):
        """
        Parameters
        ----------
        n_inv : array_like
            Inverse noise map, (npix, ) or (2, npix). In the latter case, the two rows represent NET and NEQ/U
        tf: TFObj
        g: Geometry
            ducc wrapper to speed up alm2map/map2alm
        """

        _ninv = np.atleast_2d(n_inv)
        assert _ninv.shape[0] in [1, 2, 3]

        self.tf = tf
        self.lmax = tf.lmax
        self.almsize = hp.Alm.getsize(self.lmax)
        self.npix = _ninv.shape[-1]
        self.nside = hp.npix2nside(self.npix)
        self.pixarea = hp.nside2pixarea(self.nside)

        self.n_inv_t = None
        self.n_inv_q = None
        self.n_inv_u = None

        self.npol = tf.npol
        assert self.npol in [1, 2, 3]
        assert self.npol == _ninv.shape[0], f"ninv maps should match npol={self.npol}, got {_ninv.shape[0]}"
        if self.npol in [1, 3]:
            logger.warning(f"input ninv has shape {_ninv.shape}, taking the first row as T")
            self.n_inv_t = _ninv[0]
            self.nlev_cl_t, fsky, NET = self.ninv2nlev(self.n_inv_t)
        if self.npol in [2, 3]:
            logger.warning(f"input ninv has shape {_ninv.shape}, taking the second to last as Q")
            self.n_inv_q = _ninv[-2]
            logger.warning(f"input ninv has shape {_ninv.shape}, taking the last row as U")
            self.n_inv_u = _ninv[-1]
            self.nlev_cl_p, fsky, NET = self.ninv2nlev((self.n_inv_q+self.n_inv_u)/2)

        if g is None:
            self.g = None
        else:
            self.g = g
            assert self.g.nside == self.nside

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
            logger.error("temporarily using fixed fsky")
        nlev = 1 / np.sum(ninv) * 4 * np.pi * fsky
        NET = np.rad2deg(np.sqrt(nlev)) * 60
        logger.info(f"ninv2nlev: {NET:.2f} uK-amin noise Cl over fsky {fsky:.2f}")
        return nlev, fsky, NET

    def calc_prep(self, maps):
        maps_copy = np.copy(maps)
        self.apply_map(maps_copy)
        maps_copy /= self.pixarea
        if self.g is None:
            alms = map2alm(maps_copy, lmax=self.lmax, iter=0)
        else:
            alms = self.g.map2alm(maps_copy, lmax=self.lmax, iter=0)
        self.tf.apply_tf(alms)
        return alms

    def apply_map(self, maps):
        """map-based Ninv operation: N^-1"""
        if maps.ndim == 1:
            maps *= self.n_inv_t
        else:
            if maps.shape[0] in (1, 3):
                maps[0] *= self.n_inv_t
            if maps.shape[0] in (2, 3):
                maps[-2] *= self.n_inv_q
                maps[-1] *= self.n_inv_u

    def apply_alm(self, alms):
        """harmonic-space Ninv operation: apply A^T N^-1 A on alms"""
        assert alms.shape[-1] == self.almsize
        self.tf.apply_tf(alms)
        if self.g is None:
            maps = alm2map(alms, self.nside)
        else:
            maps = self.g.alm2map(alms)

        self.apply_map(maps)
        maps /= self.pixarea

        if self.g is None:
            alms[:] = map2alm(maps, lmax=self.lmax, iter=0)
        else:
            self.g.map2alm(maps, lmax=self.lmax, iter=0, alms=alms)
        self.tf.apply_tf(alms)
