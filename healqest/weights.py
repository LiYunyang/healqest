import logging
import numpy as np

logger = logging.getLogger(__name__)


class weights_plus:
    def __init__(self, est, cls, lmax, curl=False, u=None, distortion='lens'):
        """Init the weights in `cmblensplus` style.

        Parameters
        ----------
        est: str
            estimator name 'TT'/'TE'/'EE', etc.
        cls: dict
            CMB spectra for weights, with keys 'tt','te','ee','bb'.
        lmax: int
            lmax of Cls for weights
        u: array
            f(ell) that describe the power spectrum of a foreground or beam function for profile hardening
            (can be array of 1s)
        distortion: str
            type of distortion, 'lens', 'amp', or 'src' (source hardening)
        """
        if est == 'prf':
            logger.warning('prf will be a distortion type rather than an estimator name in the future.')
            est = 'TT'
            distortion = 'prf'

        if distortion == 'prf':
            assert u is not None, "Must provide u(ell)"
        if curl and distortion != 'lens':
            logger.warning(f"{distortion} type doesn't has curl mode, ignore `curl`")

        self.lmax = lmax
        self.distortion = distortion
        logger.debug('Computing weights -- cmblensplus style')

        sl = {ii: cls[ii] for ii in cls.keys()}

        self.w = dict()
        self.s = dict()
        self.ntrm = None
        self.l = np.arange(self.lmax + 1, dtype=np.float64)

        if est in ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']:
            self.init_weights(est, sl=sl, curl=curl, u=u)
        elif est in ['ET', 'BT', 'BE']:
            self.init_weights(est[::-1], sl=sl, swap=True, curl=curl, u=u)
        # elif est in ['TTcurl', 'EEcurl', 'BBcurl', 'TEcurl', 'TBcurl', 'EBcurl', 'T2curl', 'T1curl']:
        #     self.init_weights(est[:2], sl=sl, curl=True)
        # elif est in ['ETcurl', 'BTcurl', 'BEcurl']:
        #     self.init_weights(est[:2][::-1], sl=sl, swap=True, curl=True)
        else:
            raise NotImplementedError(f"{est} is not implemented yet")

    @classmethod
    def estimators(cls, distortion):
        if distortion == 'lens':
            return ['TT', 'TE', 'TB', 'EE', 'EB', 'ET', 'BT', 'BE']
        elif distortion == 'tau':
            return ['TT', 'TE', 'TB', 'EE', 'EB', 'ET', 'BT', 'BE']
        elif distortion == 'rot':
            return ['TE', 'TB', 'EE', 'EB', 'ET', 'BT', 'BE']
        elif distortion == 'prf':
            return ['TT']
        else:
            raise NotImplementedError(f"{distortion} is not implemented yet")

    def add_term(self, ws1, ws2):
        idx = len(self.w)
        w1, s1 = ws1
        w2, s2 = ws2
        self.w[idx] = {0: w1, 1: w2}
        self.s[idx] = {0: s1, 1: s2}

    def w_X0(self, X, conj=False):
        r"""Coefficients for spin 0 fields \bar{Theta}, \bar{P}E and \bar{P}B in cmblensplus document."""
        if X == 'T':
            w = np.ones_like(self.l)
            s = 0
        elif X == 'E':
            w = -np.ones_like(self.l)
            s = 2
        elif X == 'B':
            w = -1j * np.ones_like(self.l)
            s = 2
        else:
            raise ValueError("X must be 'T', 'E' or 'B'")
        if conj:
            s = -s
            w = np.conj(w) * (-1) ** s
        return w, s

    def w_X0j(self, X, conj=False):
        # like w_X0, just use j for E instead of B.
        if X == 'E':
            w = -1j * np.ones_like(self.l)
            s = 2
        elif X == 'B':
            w = -np.ones_like(self.l)
            s = 2
        else:
            raise ValueError("X must be 'E' or 'B'")
        if conj:
            s = -s
            w = np.conj(w) * (-1) ** s
        return w, s

    def w_X01(self, cl, s):
        """Coefficients for spin 0+1 fields and spin 2+-1 fields (eq 169/170)."""
        w = cl[: self.lmax + 1].copy()
        if s in [-1, 1]:
            w *= -np.sqrt(self.l * (self.l + 1))
            if s < 0:
                w *= (-1) ** s
        return w, s

    def w_X21(self, cl, s, factor: complex = 1):
        """Terms corresponding to spin 2+1 fields. (eq 171)."""
        if np.abs(s) == 1:
            f = np.nan_to_num(np.sqrt((self.l + 2) * (self.l - 1)))
        elif np.abs(s) == 3:
            f = np.nan_to_num(np.sqrt((self.l + 3) * (self.l - 2)))
        else:
            raise ValueError("|s| must be 1 or 3")
        w = -f * cl[: self.lmax + 1] * factor
        if s < 0:
            w = np.conj(w) * (-1) ** s
        return w, s

    def w_X00(self, cl, s, conj=False, factor=1.0):
        """Coefficients for patchy-tau."""
        w = cl[: self.lmax + 1].copy() * factor
        if conj:
            s = -s
            w = np.conj(w) * (-1) ** s
        return w, s

    def _set_base_weights_lens(self, est, sl, *args):
        if est == 'TT':
            self.add_term(self.w_X0('T', conj=False), self.w_X01(sl['tt'], 1))
            self.add_term(self.w_X01(sl['tt'], 1), self.w_X0('T', conj=False))  # swap 1-2
        elif est == 'TE':
            self.add_term(self.w_X0('T', conj=False), self.w_X01(sl['te'], 1))
            self.add_term(self.w_X21(sl['te'], -1, factor=0.5), self.w_X0('E', conj=False))
            self.add_term(self.w_X21(sl['te'], 3, factor=-0.5), self.w_X0('E', conj=True))
        elif est == 'TB':
            self.add_term(self.w_X21(sl['te'], -1, factor=0.5), self.w_X0('B', conj=False))
            self.add_term(self.w_X21(sl['te'], 3, factor=-0.5), self.w_X0('B', conj=True))
        elif est == 'EE':
            self.add_term(self.w_X0('E', conj=False), self.w_X21(sl['ee'], -1, factor=0.5))
            self.add_term(self.w_X0('E', conj=True), self.w_X21(sl['ee'], 3, factor=-0.5))
            self.add_term(self.w_X21(sl['ee'], -1, factor=0.5), self.w_X0('E', conj=False))  # swap 1-2
            self.add_term(self.w_X21(sl['ee'], 3, factor=-0.5), self.w_X0('E', conj=True))  # swap 1-2
        elif est == 'BB':
            pass
            # if 'bb' in sl:
            #     self.add_term(self.w_X0('B', conj=False), self.w_X21(sl['bb'], -1, factor=0.5j))
            #     self.add_term(self.w_X0('B', conj=True), self.w_X21(sl['bb'], 3, factor=-0.5j))
            #     self.add_term(self.w_X21(sl['bb'], -1, factor=1j), self.w_X0('B', conj=False), )  # swap 1-2
            #     self.add_term(self.w_X21(sl['bb'], 3, factor=-1j), self.w_X0('B', conj=True), )  # swap 1-2
        elif est == 'EB':
            self.add_term(self.w_X21(sl['ee'], -1, factor=0.5), self.w_X0('B', conj=False))
            self.add_term(self.w_X21(sl['ee'], 3, factor=-0.5), self.w_X0('B', conj=True))
            if 'bb' in sl:
                self.add_term(self.w_X0('E', conj=False), self.w_X21(sl['bb'], -1, factor=0.5j))
                self.add_term(self.w_X0('E', conj=True), self.w_X21(sl['bb'], 3, factor=-0.5j))
        else:
            raise NotImplementedError

    def _set_base_weights_tau(self, est, sl, *args):
        if est == 'TT':
            self.add_term(self.w_X0('T', conj=False), self.w_X00(sl['tt'], 0))
            self.add_term(self.w_X00(sl['tt'], 0), self.w_X0('T', conj=False))  # swap 1-2
        elif est == 'TE':
            self.add_term(self.w_X0('T', conj=False), self.w_X00(sl['te'], 0))
            self.add_term(self.w_X00(sl['te'], -2, conj=False, factor=-0.5), self.w_X0('E', conj=False))
            self.add_term(self.w_X00(sl['te'], -2, conj=True, factor=-0.5), self.w_X0('E', conj=True))
        elif est == 'TB':
            self.add_term(self.w_X00(sl['te'], -2, conj=False, factor=-0.5), self.w_X0('B', conj=False))
            self.add_term(self.w_X00(sl['te'], -2, conj=True, factor=-0.5), self.w_X0('B', conj=True))
        elif est == 'EE':
            self.add_term(self.w_X0('E', conj=False), self.w_X00(sl['ee'], -2, factor=-0.5))
            self.add_term(self.w_X0('E', conj=True), self.w_X00(sl['ee'], -2, factor=-0.5, conj=True))
            self.add_term(self.w_X00(sl['ee'], -2, factor=-0.5), self.w_X0('E', conj=False))  # swap 1-2
            self.add_term(self.w_X00(sl['ee'], -2, factor=-0.5, conj=True), self.w_X0('E', conj=True))
            # swap 1-2
        elif est == 'EB':
            self.add_term(self.w_X00(sl['ee'], -2, factor=-0.5), self.w_X0('B', conj=False))
            self.add_term(self.w_X00(sl['ee'], -2, factor=-0.5, conj=True), self.w_X0('B', conj=True))
        else:
            raise NotImplementedError

    def _set_base_weights_rot(self, est, sl, *args):
        if est == 'TE':
            self.add_term(self.w_X00(sl['te'], -2, conj=False, factor=1), self.w_X0j('E', conj=False))
            self.add_term(self.w_X00(sl['te'], -2, conj=True, factor=1), self.w_X0j('E', conj=True))
        elif est == 'TB':
            self.add_term(self.w_X00(sl['te'], -2, conj=False, factor=-1), self.w_X0j('B', conj=False))
            self.add_term(self.w_X00(sl['te'], -2, conj=True, factor=-1), self.w_X0j('B', conj=True))
        elif est == 'EE':
            self.add_term(self.w_X0j('E', conj=False), self.w_X00(sl['ee'], -2, conj=False, factor=1))
            self.add_term(self.w_X0j('E', conj=True), self.w_X00(sl['ee'], -2, conj=True, factor=1))
            self.add_term(self.w_X00(sl['ee'], -2, conj=False, factor=1), self.w_X0j('E', conj=False))
            self.add_term(self.w_X00(sl['ee'], -2, conj=True, factor=1), self.w_X0j('E', conj=True))
        elif est == 'EB':
            self.add_term(self.w_X00(sl['ee'], -2, conj=False, factor=-1), self.w_X0j('B', conj=False))
            self.add_term(self.w_X00(sl['ee'], -2, conj=True, factor=-1), self.w_X0j('B', conj=True))
        else:
            raise NotImplementedError

    def _set_base_weights_prf(self, est, sl, u):
        if est == 'TT':
            self.add_term(ws1=(u[: self.lmax + 1], 0), ws2=(u[: self.lmax + 1], 0))
        else:
            raise NotImplementedError

    def set_base_weights(self, distortion, est, sl, u):
        if est not in self.__class__.estimators(distortion):
            raise ValueError('distortion %s is not a valid estimator' % distortion)
        if distortion == 'lens':
            func = self._set_base_weights_lens
        elif distortion == 'tau':
            func = self._set_base_weights_tau
        elif distortion == 'rot':
            func = self._set_base_weights_rot
        elif distortion == 'prf':
            func = self._set_base_weights_prf
        else:
            raise NotImplementedError(f"{distortion} is not implemented yet")
        try:
            func(est, sl, u)
        except NotImplementedError:
            raise NotImplementedError(f"{est} is not implemented yet for {self.distortion} field")

    def init_weights(self, est, sl, swap=False, curl=False, u=None):
        """Initialize weights for the base estimators: TT/EE/BB/TE/TB/EB.

        Parameters
        ----------
        est: str
        sl: dict
            with keys 'tt','te','ee' ('bb')
        swap: bool=False
            swap the weights for map1 and map2, so TE -> ET
        curl: bool=False
            modify the weights such that "grad" gives the curl estimator and vice versa (off by -1).
        u: np.ndarray
            the profile function for profile hardening.
        """
        self.set_base_weights(distortion=self.distortion, est=est, sl=sl, u=u)

        if swap:
            # flip the weights for asymmetric estimators
            for k, w in self.w.items():
                s = self.s[k]
                self.w[k][0], self.w[k][1] = w[1], w[0]
                self.s[k][0], self.s[k][1] = s[1], s[0]

        if self.distortion == 'lens':
            # non-hardened estimators
            f3 = np.sqrt(self.l * (self.l + 1)) * 0.5
            # YL doesn't understand the 0.5 factor. But this is consistent with old implementation
            # (also doesn't matter)
            if curl:
                f3 = -f3 * 1j
            s3 = 1
        elif self.distortion in ['tau', 'rot']:
            f3 = np.ones(self.lmax + 1) * 0.5
            s3 = 0
        elif self.distortion == 'prf':
            # profile hardened estimators for lensing
            f3 = 1 / u[: self.lmax + 1] * 0.5  # the 0.5 factor accounts for the second half redundant terms.
            s3 = 0
        else:
            raise NotImplementedError(f"{self.distortion} is not implemented yet")

        for k, w in self.w.items():
            self.w[k][2] = f3
            self.s[k][2] = s3

        # adding the second half "redundant" terms.
        ntrm = len(self.w)
        new_w = dict()
        new_s = dict()
        for k, w in self.w.items():
            s = self.s[k]
            new_w[k + ntrm] = dict()
            new_s[k + ntrm] = dict()
            for i in range(3):
                # for w[2], the second half has a -1/1 factor for grad
                # and curl mode. Since s[2] is always [1, -1], it happens
                # to be the case that the required sign change is also consistent
                # with conj(w[2])*(-1)**s[2], where w[2] is pure imag for curl
                # modes.
                new_w[k + ntrm][i] = np.conj(w[i]) * (-1) ** s[i]
                new_s[k + ntrm][i] = -s[i]

        self.w.update(new_w)
        self.s.update(new_s)
        self.ntrm = len(self.w)
