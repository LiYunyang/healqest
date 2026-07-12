from typing import TypedDict, Callable, Optional

import numpy as np
import healpy as hp
from healqest import healqest_utils as utils
from healqest import weights, resp, log, ducc_sht
from functools import lru_cache, partial

logger = log.get_logger(__name__)
np.seterr(all='ignore')


class CMBCl(TypedDict):
    tt: np.ndarray
    te: np.ndarray
    ee: np.ndarray
    bb: np.ndarray


class Qest:
    """QE estimator following the cmblensplus convention."""

    __PH_ESTIMATORS__ = ['TTph', 'EEph', 'TEph', 'ETph']
    # allowed single harden estimators, excluding the odd parity ones. The "GMVph" etc is made from these.

    harden_cache: Optional['HardenCache'] = None

    def __init__(self, lmax, Lmax, Cls, fls, fls2=None, g=None, flT=None, flP=None, fast=True):
        """Setup quadratic estimator.

        Parameters
        ----------
        lmax: int
            Maximum multipole of the cmb map alm
        Lmax: int
            Maximum multipole of the lens rec alm
        Cls: CMBCl
            dict of cls
        fls, fls2: np.ndarray.
            filter functions of shape (4, lmax+1) for TT/EE/BB/TE, for the inverse-Cl matrix elements (for
            SQE, this should be the same as 1/Cl, but in gmv case, the TE covariance is properly handled).
            These will be converted internallly into a dict.
        g: Geometry
            Geometry instance defined within declination range. This will be used to compute spherical
            harmonics functions with ducc0. If None, the slower full-sky healpy functions will be used.
        flT, flP: np.ndarray=None
            binary array of shape (lmax+1, ), indicating the POST cinv ell selection.
        fast: bool=False
            If True, uses the fast response function calculation.
        """
        self.lmax = lmax
        self.Lmax = Lmax
        self.size = hp.Alm.getsize(self.lmax)

        self.fls = self.fls2dict(fls)
        if fls2 is not None:
            self.fls2 = self.fls2dict(fls2)
        else:
            self.fls2 = self.fls
        assert np.any(self.fls['TE'] != 0) == np.any(self.fls2['TE'] != 0), (
            "TE filter must be either both non-zero or both zero for fls and fls2"
        )
        self.gmv = np.any(self.fls['TE'] != 0) and np.any(self.fls2['TE'] != 0)

        # Post cinv ell cut
        self.fl_cut = dict()
        self.fl_cut['T'] = flT[: self.lmax + 1] if flT is not None else np.ones(self.lmax + 1)
        self.fl_cut['E'] = flP[: self.lmax + 1] if flP is not None else np.ones(self.lmax + 1)
        self.fl_cut['B'] = flP[: self.lmax + 1] if flP is not None else np.ones(self.lmax + 1)
        assert self.fl_cut['T'].shape[-1] == self.lmax + 1
        assert self.fl_cut['E'].shape[-1] == self.lmax + 1
        assert self.fl_cut['B'].shape[-1] == self.lmax + 1

        self.cls = Cls
        if g is None:
            nside = utils.get_nside(lmax)
            g = ducc_sht.Geometry(nside=nside)
        self.g = g
        self.nside = g.nside
        assert self.lmax < 2.0 * self.nside, "lmax must be less that 2*nside"
        self.fast = fast

        self.slm_cache = None

    def init_harden(self, u: np.ndarray, almbars1=None, almbars2=None):
        """
        Prepare the profile hardening data (weights, response, and template).

        Parameters
        ----------
        u: np.ndarray
            profile functions for TTph estimator.
        almbars1, almbars2: complex arrays, optional.
            The data to reconstruct harden template.
        """
        self.harden_cache = HardenCache(self, u=np.atleast_2d(u))
        if almbars1 is not None:
            if not np.allclose(almbars1, almbars2):
                logger.warning("Using diff legs for profile hardening is ill-defined. Be cautious.")
            key = (id(almbars1), id(almbars2))
            self.slm_cache = {key: []}
            for j, _u in enumerate(np.atleast_2d(u)):
                _slm = self.eval('TT', almbars1[0], almbars2[0], u=_u, distortion='prf')[0]
                self.slm_cache[key].append(_slm)

    @staticmethod
    def alm2map_spin(alm, fell, nside, spin, lmax, mmax=None, g=None):
        """Convert a spin-0 alm into a complex spin field (Q +/- iU): out = Q, +/-U."""
        if spin == 0:
            walm = hp.almxfl(alm, fell)
            # alm2map is recommended over alm2map_spin for spin=0
            if g is None:
                out = hp.alm2map(walm, nside=nside, lmax=lmax, mmax=mmax)
            else:
                out = g.alm2map(walm, lmax=lmax, mmax=mmax)
            return out, 0
        else:
            zero = np.zeros_like(alm)
            _fell = (-1) ** spin * np.conj(fell) if spin < 0 else fell
            _fell *= -1
            if np.all(fell.imag == 0):
                E = hp.almxfl(alm, _fell.real)
                B = zero
            elif np.all(fell.real == 0):
                E = zero
                B = hp.almxfl(alm, _fell.imag)
            else:
                raise ValueError("Fell must be real or imaginary")
            if g is None:
                q, u = hp.alm2map_spin([E, B], nside=nside, spin=np.abs(spin), lmax=lmax, mmax=mmax)
            else:
                q, u = g.alm2map_spin([E, B], spin=np.abs(spin), lmax=lmax, mmax=mmax)
            if spin > 0:
                return q, u
            else:
                return q, -u

    def eval(self, qe, almbar1, almbar2, u=None, distortion='lens'):
        """Compute quadratic estimator.

        Parameters
        ----------
        qe: str
          Quadratic estimator type (defined in `weights_plus`): 'TT'/'EE'/'TE'/'EB'/'TB'
        almbar1,almbar2: complex array healpy alm
          First and second filtered alm
        u: np.ndarray=None
          Profile instance
        distortion: str
            distortion type, 'lens', 'rot', 'prf' or 'tau'

        Returns
        -------
        glm, clm: tuple of complex array
            Gradient/curl component of the plm
        """
        assert almbar1.shape[-1] == self.size, (
            f"almbar size {almbar1.shape[-1]} don't match lmax {self.lmax})"
        )
        assert almbar2.shape[-1] == self.size, (
            f"almbar size {almbar2.shape[-1]} don't match lmax {self.lmax})"
        )

        if distortion in ['prf']:
            assert u is not None, "Need profile function to compute this estimator"

        q = weights.weights_plus(qe, self.cls, self.lmax, u=u, distortion=distortion)

        logger.info(f'Running {distortion} reconstruction: {qe}')

        retglm = 0
        retclm = 0

        assert q.ntrm % 2 == 0, f"Number of terms must be even: {q.ntrm}"
        for i in range(0, q.ntrm // 2):
            # skipping second half of reducant terms
            wX, wY, wP, sX, sY, sP = q.w[i][0], q.w[i][1], q.w[i][2], q.s[i][0], q.s[i][1], q.s[i][2]

            Xq, Xu = self.alm2map_spin(almbar1, fell=wX, nside=self.nside, spin=sX, lmax=self.lmax, g=self.g)
            Yq, Yu = self.alm2map_spin(almbar2, fell=wY, nside=self.nside, spin=sY, lmax=self.lmax, g=self.g)
            XYq = Xq * Yq - Xu * Yu  # XY = X*Y
            XYu = Xq * Yu + Yq * Xu  # XY = X*Y

            if np.all(wP.imag == 0):
                _wP = wP
            elif np.all(wP.real == 0):
                # swap grad/curl mode such that glm is curl and clm is grad
                # wP has an -1j factor, here we move the factor from wP to XY.
                _wP = wP * 1j
                XYq, XYu = XYu, -XYq  # XY *=-1j
            else:
                raise ValueError("wP must be real or imaginary")
            if sP < 0:
                # This is for the second half reduncant transform, we normally don't end up here.
                # XY = np.conj(XY) * (-1) ** sP  # because wP has a (-1)**sP factor, here we are canceling it.
                XYq *= (-1) ** sP  # XY = np.conj(XY) * (-1) ** sP
                XYu *= -((-1) ** sP)  # XY = np.conj(XY) * (-1) ** sP

            glm, clm = self.g.map2alm_spin([XYq, XYu], spin=np.abs(sP), lmax=self.Lmax, check=False)
            glm = hp.almxfl(glm, _wP)
            clm = hp.almxfl(clm, _wP)  # for curl est, this will be -grad.

            retglm += glm
            retclm += clm

        return retglm, retclm

    @staticmethod
    def fls2dict(fls: np.ndarray):
        out = dict(zip(['TT', 'EE', 'BB', 'TE'], fls))
        out['ET'] = fls[3]
        return out

    def get_resp(self, qe, u=None, curl=False, type1='lens', type2=None, u2=None):
        r"""
        Compute the cross response between two estimators. Assume joint cinv filtering.

        Note
        ----
        For example, we want to see how much the f^XY estimator extract the distrotion field (encoded by
        weights g) from filtered maps \bar{X} and \bar{Y}, which can be decomposed into W^{XZ} Z and W^{YA} A,
        where X/Y/Z/A are T/E/B, and W are the filter functions (gmv, or sqe in the diagonal case).
        In the general form, this is computing sum_{ZA} f^{XY} W^{XZ} W^{YA} g^{ZA}. Note that, some terms
        in `g` might not exist and will be skipped, e.g. non-TT terms for the profile estimator.

        Parameters
        ----------
        qe: str
            Quadratic estimator type, e.g., 'TT','EB'
        u, u2: np.ndarray=None
            profile function for prf estimator
        curl: bool
            If True, `qe` is suffixed with `curl` to compute curl-mode response.
        type1, type2: str
            distortion field  type for the estimator, 'lens' or 'prf' or 'tau' or 'rot'.

        Returns
        -------
        R: np.ndarray
            response function
        """
        R = np.zeros(self.Lmax + 1, dtype=float)
        if qe not in weights.weights_plus.estimators(type1):
            logger.warning(f"{type1} distortion does not have {qe} defined. set response to 0.")
            return R
        else:
            qeXY = weights.weights_plus(qe, self.cls, self.lmax, distortion=type1, curl=curl, u=u)

        if type2 is None:
            type2 = type1

        s1, s2 = qe[0], qe[1]
        assert s1 in 'TEB' and s2 in 'TEB', f"qe must be one of TEB, got: {qe}"

        if self.gmv:
            keys = list(self.fls.keys())
        else:
            keys = [f"{s1}{s1}", f"{s2}{s2}"]  # SQE only picks the 2 (can be the same) diagonal terms.

        for qe2 in [s1 + s2 for s1 in 'TEB' for s2 in 'TEB']:
            if qe2 not in weights.weights_plus.estimators(type2):
                # sometimes `_qe2` is not defined for the second distortion field,
                # in this case it should be skipped.
                continue
            k1 = s1 + qe2[0]
            k2 = s2 + qe2[1]
            if k1 not in keys or k2 not in keys:
                # the weights of these combination are zero, so skip as well.
                continue
            flX = self.fls[k1] * self.fl_cut[s1]
            flY = self.fls2[k2] * self.fl_cut[s2]
            qeZA = weights.weights_plus(qe2, self.cls, self.lmax, distortion=type2, curl=curl, u=u2)
            R += resp.fill_resp_fullsky(
                qeXY, qeZA, np.zeros(self.Lmax + 1, dtype=complex), flX, flY, fast=self.fast
            )
        return R

    def rec_and_resp(self, qe, almbars1, almbars2, type1='lens', compute_resp=True):
        """
        Compute lensing reconstruction for grad and curl modes, return also the analytical response functions.

        Parameters
        ----------
        qe: str
            Quadratic estimator type, e.g., 'TT','TTph'
        almbars1, almbars2: complex arrays
            First and second filtered alms, shape (3, nalm)
        type1: str
            distortion field  type for the estimator, 'lens' or 'prf' or 'tau'
        compute_resp: bool
            If True, compute the analytical response function for grad/curl mode.

        Returns
        -------
        [glm, clm]: list of complex array
            Gradient/curl component of the plm
        [aresp_g, aresp_c]: list of np.ndarray
            Analytical response function for grad/curl mode
        hrd_out: dict or None
            If `qe` ends with 'ph', return a dict containing the source response functions
        """
        i1 = 'teb'.index(qe[0].lower())
        i2 = 'teb'.index(qe[1].lower())

        if qe.endswith('ph'):
            _qe = qe.removesuffix('ph')
        else:
            _qe = qe
        if almbars1 is not None:
            glm, clm = self.eval(_qe, almbars1[i1], almbars2[i2], distortion=type1)
        else:
            glm, clm = None, None
        if compute_resp:
            aresp_g = self.get_resp(_qe, type1=type1, type2=type1)
            if type1 == 'lens':
                aresp_c = self.get_resp(_qe, curl=True, type1=type1, type2=type1)
            else:
                aresp_c = np.zeros_like(aresp_g)
        else:
            aresp_g = aresp_c = None

        if qe.endswith('ph'):
            key = (id(almbars1), id(almbars2))
            slm_cache = self.slm_cache.get(key) if self.slm_cache is not None else None
            self.profile_harden(_qe, glm, clm, aresp_g, aresp_c, slm_cache, type1=type1, harden_curl=False)
        return [glm, clm], [aresp_g, aresp_c]

    def profile_harden(self, qe: str, glm, clm, aresp_g, aresp_c, slm_cache, type1='lens', harden_curl=False):
        # do the source harden stuff
        _qe = qe.removesuffix('ph')
        assert self.harden_cache is not None, "Need HardenCache to compute this estimator"
        if not self.gmv:
            assert _qe == 'TT', f"We only harden for 'TT' for SQE, got: {qe}"
        for j, _u in enumerate(self.harden_cache.u):
            _w = self.harden_cache.get_harden_weights(_qe, j, curl=False, type1=type1)
            if glm is not None:
                glm += hp.almxfl(slm_cache[j], _w)
            if aresp_g is not None:
                _r = self.harden_cache.get_harden_response(j, curl=False, type1=type1)
                aresp_g += _r * _w
            if type1 == 'lens' and harden_curl:
                _w = self.harden_cache.get_harden_weights(_qe, j, curl=True, type1=type1)
                if clm is not None:
                    clm += hp.almxfl(slm_cache[j], _w)
                if aresp_c is not None:
                    _r = self.harden_cache.get_harden_response(j, curl=True, type1=type1)
                    aresp_c += _r * _w
        return glm, clm, aresp_g, aresp_c


class HardenCache:
    """Caches for profile-harden weights and responses.

    Notes
    -----
    The key of profile hardening is to compute the following matrix (R)
        | R^{pp}  R^{p1} R^{p2} |
        | R^{1p}  R^{11} R^{12} |
        | R^{2p}  R^{21} R^{22} |
     and its cofactors (∆).
    """

    def __init__(self, estimator: Qest, u: np.ndarray):
        self.qest = estimator
        self.u = np.atleast_2d(u)
        self.nprf = self.u.shape[0]
        self.full_seq = list(np.arange(self.nprf + 1))  # full index sequence of the (n+1, n+1) matrix.

        # this is the common denominator for all the hardening weights, and it is based on profiles only, so
        # it should be independent of the QE type or curl/grad modes.
        func = partial(self.get_R, curl=False, qe=None)
        self.D00 = self.cofactor(self.full_seq, self.full_seq, 0, 0, func)

    def get_R(self, i: int, j: int, qe: Optional[str] = None, type1='lens', curl: bool = False):
        """
        Compute the (cached) response matrix element R_{ij} for profile hardening.

        Notes
        -----
        Note that this is not the actual symmetric response matrix. For the phi-source terms in the first row,
        R0j is per-QE type, i.e., R_0j^{XY}, but the transpose terms Ri0 is summed over all QE types,
        i.e., R_i0 = sum_{XY} R_i0^{XY}. This is for the convinence of computing individual GMV terms.
        See also https://sptlocal.grid.uchicago.edu/~yyli/SPT_lensing/multi_profile_hardening/index.html.

        Parameters
        ----------
        i,j: int
            Index of the response matrix (n+1, n+1). The main QE estimator is (0, 0).
        qe: str, optional
            Quadratic estimator type, e.g., 'TT','EB'. Only needed for the first row (i=0, R00 and R0j).
        type1: str
            The main distortion field type, 'lens' or 'tau' or 'rot'.
        curl: bool
            If True, compute the curl-mode response (for lensing).
        """
        assert i in range(self.nprf + 1)
        assert j in range(self.nprf + 1)

        if i == j == 0:
            # phi-phi (lensing response)
            assert qe is not None
            return self._get_R00(qe=qe.removesuffix('ph'), type1=type1, curl=curl)
        elif i == 0:
            # phi-source terms, used to determine weights (i.e. how much source contribute to lensing)
            assert qe is not None
            return self._get_R0j(j, qe=qe.removesuffix('ph'), type1=type1, curl=curl)
        elif j == 0:
            # source-phi terms, used only for response (i.e., how much the source over subtracts lensing)
            # note that this is assymetric and has additional summation over lensing QE types.
            return self._get_Ri0(i, type1=type1, curl=curl)
        else:
            # source-source terms
            return self._get_Rij(i, j)

    @lru_cache(maxsize=16)  # good for upto 4 profiles
    def _get_Rij(self, i: int, j: int):
        return self.qest.get_resp(
            'TT', type1='prf', type2='prf', u=self.u[i - 1], curl=False, u2=self.u[j - 1]
        )

    @lru_cache(maxsize=8)  # good for upto 4 profiles
    def _get_Ri0(self, i: int, type1: str, curl: bool):
        return self.qest.get_resp('TT', type1='prf', type2=type1, u=self.u[i - 1], curl=curl)

    @lru_cache(maxsize=72)  # good for upto 4 profiles
    def _get_R0j(self, j: int, qe: str, type1: str, curl: bool):
        return self.qest.get_resp(qe, type1=type1, type2='prf', u2=self.u[j - 1], curl=curl)

    @lru_cache(maxsize=18)
    def _get_R00(self, qe: str, type1: str, curl: bool):
        return self.qest.get_resp(qe, type1=type1, type2=type1, curl=curl)

    def get_harden_weights(self, qe: str, j: int, curl: bool = False, type1: str = 'lens'):
        assert j in range(self.nprf)
        func = partial(self.get_R, curl=curl, qe=qe, type1=type1)
        Ck = self.cofactor(self.full_seq, self.full_seq, j + 1, 0, func)
        return Ck / self.D00

    @lru_cache(maxsize=10)
    def get_harden_response(self, j: int, curl: bool = False, type1: str = 'lens'):
        assert j in range(self.nprf)
        return self.get_R(j + 1, 0, qe=None, curl=curl, type1=type1)

    @staticmethod
    def det(seq_i: list, seq_j: list, func: Callable):
        """
        Compute the determinant of a (sub) matrix.

        Parameters
        ----------
        seq_i: list.
            Row indices of the submatrix.
        seq_j: list.
            Column indices of the submatrix.
        func: Callable
            A function that takes two indices (ii, jj) and returns the corresponding matrix element, where
            ii and jj are the same type as the elements in seq_i and seq_j.

        Returns
        -------
        det: float
        """
        assert len(seq_i) == len(seq_j), "Row and column indices must have the same length."
        if len(seq_i) == len(seq_j) == 1:
            return func(seq_i[0], seq_j[0])
        tot = 0
        for _j, j in enumerate(seq_j):
            tot += func(seq_i[0], j) * HardenCache.cofactor(seq_i, seq_j, 0, _j, func=func)
        return tot

    @staticmethod
    def cofactor(seq_i: list, seq_j: list, ki: int, kj: int, func: Callable):
        """
        Compute the cofactor of a matrix element at position (ki, kj) in the submatrix defined by seq_i/seq_j.

        Parameters
        ----------
        seq_i: list.
            Row indices of the submatrix.
        seq_j: list.
            Column indices of the submatrix.
        ki, kj: int
            Row and column indices of the element for which the cofactor is computed, relative to seq_i/j.
        func: Callable
            A function that takes two indices (ii, jj) and returns the corresponding matrix element, where
            ii and jj are the same type as the elements in seq_i and seq_j.
        """
        # sometimes we run into computing cofactor of a 1x1 matrix.
        if len(seq_i) == len(seq_j) == 1:
            return np.ones_like(func(seq_i[0], seq_j[0]))
        sign = (-1) ** (ki + kj)
        i_sub = list(seq_i.copy())
        j_sub = list(seq_j.copy())
        i_sub.pop(ki)
        j_sub.pop(kj)
        det_minor = HardenCache.det(i_sub, j_sub, func)
        return sign * det_minor
