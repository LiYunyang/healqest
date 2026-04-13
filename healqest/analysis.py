from functools import partial
import numpy as np
import multiprocessing
from scipy.signal import savgol_filter
import logging
import operator


def bin_spectrum(Cls, bins, *, return_error=False, verbose=True, weight=False):
    """
    The returned error is the "error of the mean".

    Parameters
    ----------
    Cls: np.ndarray(nspec, nstokes, nell) or
        np.ndarray(nstokes, nell) or np.ndarray(nstokes, )
        The Cls to be binned.
    bins: np.array(nbin+1, )
        The bins used. The left edge is included while the right edge is Excluded.
    weight: bool=False
        If True, use the l(l+1) weights on the Cl.
    return_error: bool=False
        If True, return the error.
    verbose: bool=True

    Returns
    -------
    ellb: np.ndarray(nbin)
    Clsb: np.ndarray(nstokes, nbin)
    """
    if np.array(Cls).ndim == 3:
        _Cls = np.transpose(Cls, (1, 2, 0))
    else:
        _Cls = np.atleast_3d(Cls)
    nstoke, nell, nspec = _Cls.shape
    if verbose:
        print(f'nstoke={nstoke}, nell={nell}, nspec={nspec}')

    ell = np.arange(nell)
    if weight:
        fac = 2 * ell + 1
    else:
        fac = np.ones_like(ell)
    _Cls = np.einsum('ijk,j->ijk', _Cls, fac)

    bin_idx = np.digitize(ell, bins, right=False)
    bin_norm = np.bincount(bin_idx, weights=fac)

    ellb = np.bincount(bin_idx, ell * fac) / bin_norm
    Clb = np.array([np.bincount(bin_idx, np.mean(_, axis=-1)) for _ in _Cls]) / bin_norm
    slc = slice(1, len(bins))
    if return_error:
        Clb_expand = Clb[:, bin_idx].reshape(nstoke, nell, 1) * fac[None, :, None]
        Clb_err = (
            np.sqrt(np.array([np.bincount(bin_idx, np.sum(_**2, axis=-1)) for _ in _Cls - Clb_expand]))
            / bin_norm
            / nspec
        )
        return ellb[slc], np.squeeze(Clb[:, slc]), np.squeeze(Clb_err[:, slc])
    else:
        return ellb[slc], np.squeeze(Clb[:, slc])


def bin_Cls(Cls, bins):
    """Return bin center, bined Cls, error on the mean and cov.

    Parameters
    ----------
    Cls: np.ndarray
        shape (nspec, nell) or (nell, )
    """
    x = (bins[1:] + bins[:-1]) / 2
    Cbs = np.array([bin_spectrum(_, bins=bins, verbose=False)[1] for _ in np.atleast_2d(Cls)])
    cov = np.cov(Cbs, rowvar=False)
    err = np.std(Cbs, axis=0) / np.sqrt(Cls.shape[0])
    return x, np.mean(Cbs, axis=0), err, cov


def load_Cls(i, config, mvtype, cross=False, average=True, coadd=False, cmbset='a', curl=False):
    """Load the data (xx-type) Cls."""
    ktype2 = None if cross else 'xx'
    fname = config.p_cls(
        tag=mvtype, seed1=i, seed2=None, ktype1='xx', ktype2=ktype2, coadd=coadd, cmbset=cmbset, curl=curl
    )
    Cls = np.loadtxt(fname)
    if average:
        return np.mean(Cls[:, 1:], axis=-1)
    else:
        return Cls[:, -1]


def load_Cls_ab(i, config, mvtype, average=True, coadd=False, curl=False):
    Cls = np.loadtxt(config.p_cls(mvtype, i, None, 'aa', 'bb', N1=True, coadd=coadd, curl=curl))
    if average:
        return np.mean(Cls[:, 1:], axis=-1)
    else:
        return Cls[:, -1]


def load_N0(i, config, mvtype, average=True, SAN0=False, coadd=False, cmbset='a', curl=False):
    if SAN0:
        Cls = np.loadtxt(config.p_cls(mvtype, i, i, 'xx', 'xx', SAN0=True, curl=curl))
        return Cls[:, -1]
    _ = 0
    Cls_xyxy = np.loadtxt(config.p_cls(mvtype, i, None, 'xy', 'xy', coadd=coadd, cmbset=cmbset, curl=curl))
    Cls_xyyx = np.loadtxt(config.p_cls(mvtype, i, None, 'xy', 'yx', coadd=coadd, cmbset=cmbset, curl=curl))
    if average:
        _ += np.mean(Cls_xyxy[:, 1:], axis=-1)
        _ += np.mean(Cls_xyyx[:, 1:], axis=-1)
    else:
        _ = Cls_xyxy[:, -1] + Cls_xyyx[:, -1]
    return _


def load_RDN0(i, config, mvtype, average=True, coadd=False, cmbset='a', curl=False):
    def load(fname):
        dat = np.loadtxt(fname)
        if average:
            return np.mean(dat[:, 1:], axis=-1)
        else:
            return dat[:, -1]

    tot = load(config.p_cls(mvtype, i, None, 'x0', 'x0', coadd=coadd, cmbset=cmbset, curl=curl))
    tot += load(config.p_cls(mvtype, i, None, 'x0', '0x', coadd=coadd, cmbset=cmbset, curl=curl))
    tot += load(config.p_cls(mvtype, i, None, '0x', '0x', coadd=coadd, cmbset=cmbset, curl=curl))
    tot += load(config.p_cls(mvtype, i, None, '0x', 'x0', coadd=coadd, cmbset=cmbset, curl=curl))
    return tot


def load_N1(i, config, mvtype, average=True, coadd=False, curl=False):
    Cls_xyxy = np.loadtxt(config.p_cls(mvtype, i, None, 'xy', 'xy', N1=True, coadd=coadd, curl=curl))
    Cls_xyyx = np.loadtxt(config.p_cls(mvtype, i, None, 'xy', 'yx', N1=True, coadd=coadd, curl=curl))
    if average:
        N0 = np.mean(Cls_xyxy[:, 1:] + Cls_xyyx[:, 1:], axis=-1)
    else:
        N0 = Cls_xyxy[:, -1] + Cls_xyyx[:, -1]
    Cls_abab = np.loadtxt(config.p_cls(mvtype, i, None, 'ab', 'ab', N1=True, coadd=coadd, curl=curl))
    Cls_abba = np.loadtxt(config.p_cls(mvtype, i, None, 'ab', 'ba', N1=True, coadd=coadd, curl=curl))
    if average:
        _ = np.mean(Cls_abab[:, 1:] + Cls_abba[:, 1:], axis=-1)
    else:
        _ = Cls_abab[:, -1] + Cls_abba[:, -1]
    return _ - N0, N0


class LensingSpectra:
    def __init__(
        self,
        config,
        N,
        mvtype,
        Lmax=None,
        resp_type='auto',
        average=True,
        N_N1=None,
        coadd=False,
        resp_smooth=None,
        cmbset='a',
        do_SAN0=False,
        do_RDN0=False,
        do_data=False,
        curl=False,
    ):
        """Lensing spectra object.

        Parameters
        ----------
        N: int
            Number of spectra to load.
        resp_type: str
            Type of MC response function. Types include `cross`,`cross2`,
            and `auto`.  `cross` and `cross2` are based on cross-correlations.
            `cross` simply takes the square of the cross-spectra response
            function, while `cross2` tries to infer the auto-correlation given
            the low-ell mode couplings, and should be more accurate at low
            multipoles. `auto` is based on the auto-spectra and should be most
            accurate at all multipoles.
        average: bool=True
            If True, average all spectra in the file, otherwise return the last
            column.
        coadd: bool
            Special case to load spectrum from `cls_coadd/` instead of `cls/`, where the lensing
            reconstruction map is coadded before taking spectra.
        """
        self.config = config
        self.cmbset = cmbset
        self.curl = curl
        self.average = average
        if Lmax is None:
            self.Lmax = self.config.Lmax
        else:
            self.Lmax = Lmax
        self.resp_type = resp_type
        self.mvtype = mvtype
        self.coadd = coadd
        self.N = N
        self.N_N1 = N_N1 if N_N1 is not None else N

        self.N0s = None  # N0 spectra, (N, Lmax+1)
        self.N1s = None  # N1 spectra, (N_N1, Lmax+1)
        self.RDN0 = None  # RDN0, (Lmax+1, )
        self.SAN0s = None  # SAN0 spectra, (N, Lmax+1)
        self.Cls_hat = None  # resp-cor, undebiased sims spectra, 1-N, shape (N, Lmax+1)
        self.Cls = None  # resp-cor, bias subtracted sims spectra, 1-N, shape (N, Lmax+1)
        self.Cl0 = None  # data spectrum, (Lmax+1,), resp corrected

        self.resp2 = None  # MC correction to response function, shape (Lmax+1, )
        self.resp2_cls = None  # raw resp spectra, shape (N_N1, Lmax+1)

        self.do_SAN0 = do_SAN0
        self.do_RDN0 = do_RDN0
        self.do_data = do_data

        self.load_resp(resp_smooth)
        self.load()

        self.x = None  # binned ell
        self.y0 = None  # binned data spectrum
        self.y = None  # binned sims spectrum
        self.cov = None  # total covariance
        self.cov_sys = None  # systematic part of the covariance
        self.yerr = None  # single rlz error, including sys and stats parts
        self.yerr_mean = None  # error on the mean. Only the stats part is reduced by sqrt(N).

    @staticmethod
    def smooth_resp(y, seq):
        out = y.copy()
        for s in seq:
            out[s:] = savgol_filter(y, window_length=s, polyorder=3)[s:]
        return out

    @property
    def fsky(self):
        return np.mean(self.config.mask_ps**2)

    @property
    def clkk(self):
        return np.loadtxt(self.config.path(self.config.clkk_in))[: self.Lmax + 1]

    @property
    def N0(self):
        return np.mean(self.N0s, axis=0)

    @property
    def N1(self):
        if self.N1s is not None:
            return np.mean(self.N1s, axis=0)
        else:
            return np.zeros(self.Lmax + 1)

    def load_resp(self, resp_smooth=None):
        if self.resp_type == 'auto':
            if self.N_N1 > 0:
                _f = partial(
                    load_Cls_ab,
                    config=self.config,
                    mvtype=self.mvtype,
                    average=self.average,
                    coadd=self.coadd,
                )
                Cls_ab = np.array(list(map(_f, range(1, self.N_N1 + 1))))[:, : self.Lmax + 1]
                self.resp2_cls = Cls_ab / self.clkk[: self.Lmax + 1]
                self.resp2 = np.mean(Cls_ab / self.clkk[: self.Lmax + 1], axis=0)
            else:
                logging.warning("not loading resp function due to no N1 sims")
                self.resp2 = np.ones(self.Lmax + 1)
                return
        elif self.resp_type in ['cross', 'cross2']:
            if self.config.nbundle is None:
                bundle_loop = [None]
            else:
                bundle_loop = np.arange(self.config.nbundle)
            self.resp2 = 0
            for b in bundle_loop:
                loaded = np.load(self.config.p_resp(self.mvtype, bundle=b))
                if self.resp_type == 'cross':
                    self.resp2 += loaded['resp'] ** 2
                elif self.resp_type == 'cross2':
                    self.resp2 += loaded['resp2']
            self.resp2 = self.resp2[: self.Lmax + 1] / len(bundle_loop)

            if self.resp_type == 'cross':
                self.resp2 *= loaded['Cl_bias'][: self.Lmax + 1]
                # which bundle doesn't matter
        else:
            logging.warning("disable resp function")
            self.resp2 = np.ones(self.Lmax + 1)
            return
        if resp_smooth is not None:
            self.resp2 = self.smooth_resp(self.resp2, np.atleast_1d(resp_smooth))

    def offload(self):
        bad_k = [k for k, v in self.config.__dict__.items() if k.startswith('mask')]
        for k in bad_k:
            del self.config.__dict__[k]

    @property
    def snr(self):
        hartlap = self.N / (self.N - self.cov.shape[0] - 1)
        # hartlap = 1
        snr = np.sqrt(np.sum(np.linalg.inv(self.cov) / hartlap))
        return snr

    def load(self):
        self.offload()
        with multiprocessing.Pool(10) as p:
            map = p.map
            if self.N_N1 > 0:
                loop = range(1, self.N_N1 + 1)
                _f = partial(
                    load_N1,
                    config=self.config,
                    mvtype=self.mvtype,
                    average=self.average,
                    coadd=self.coadd,
                    curl=self.curl,
                )
                N1s, N1N0s = np.transpose(np.array(list(map(_f, loop))), (1, 0, 2))

            loop = range(1, self.N + 1)
            _f = partial(
                load_N0,
                config=self.config,
                mvtype=self.mvtype,
                average=self.average,
                SAN0=False,
                coadd=self.coadd,
                cmbset=self.cmbset,
                curl=self.curl,
            )
            N0s = np.array(list(map(_f, loop)))

            if self.do_SAN0:
                _f = partial(
                    load_N0,
                    config=self.config,
                    mvtype=self.mvtype,
                    average=self.average,
                    SAN0=True,
                    coadd=self.coadd,
                    cmbset=self.cmbset,
                    curl=self.curl,
                )
                SAN0s = np.array(list(map(_f, loop)))

            if self.do_RDN0:
                _f = partial(
                    load_RDN0,
                    config=self.config,
                    mvtype=self.mvtype,
                    average=self.average,
                    coadd=self.coadd,
                    cmbset=self.cmbset,
                    curl=self.curl,
                )
                RDN0s = np.array(list(map(_f, loop))) - N0s

            _f = partial(
                load_Cls,
                config=self.config,
                mvtype=self.mvtype,
                average=self.average,
                coadd=self.coadd,
                cmbset=self.cmbset,
                curl=self.curl,
            )
            Cls_hat = np.array(list(map(_f, loop)))

            if self.do_data:
                Cl0 = _f(0)

        self.N0s = N0s[:, : self.Lmax + 1] / self.resp2

        if self.N_N1 > 0:
            self.N1s = N1s[:, : self.Lmax + 1] / self.resp2
        else:
            self.N1s = None

        self.Cls = Cls_hat[:, : self.Lmax + 1] / self.resp2 - self.N0 - self.N1

        if self.do_RDN0:
            self.RDN0 = np.mean(RDN0s[:, : self.Lmax + 1] / self.resp2, axis=0)

        if self.do_data:
            self.Cl0 = Cl0[: self.Lmax + 1] / self.resp2
            if self.do_RDN0:
                self.Cl0 -= self.RDN0 + self.N1
            else:
                self.Cl0 -= self.N0 + self.N1

        if self.do_SAN0:
            self.SAN0s = SAN0s[:, : self.Lmax + 1] / self.resp2

    def bin_spec(self, bins, norm_cl=None, resp_err=False):
        if norm_cl is not None:
            fac = 1 / norm_cl[: self.Lmax + 1]
        else:
            fac = 1
        self.x = (bins[1:] + bins[:-1]) / 2
        N0cov = bin_Cls(self.N0s * fac, bins)[3]
        if self.N_N1 > 0:
            N1cov = bin_Cls(self.N1s * fac, bins)[3]
        else:
            N1cov = np.diag(np.zeros(len(self.x)))

        if self.do_data:
            self.y0 = bin_Cls(self.Cl0 * fac, bins=bins)[1]
        self.y, _, self.cov = bin_Cls(self.Cls * fac, bins=bins)[1:]
        if self.do_SAN0:
            cls = self.Cls + self.N0 - self.SAN0s  # replace N0 with the per-realization SAN0
            self.cov = bin_Cls(cls * fac, bins=bins)[-1]
        self.cov_sys = N0cov / self.N + N1cov / self.N_N1  # systematic err from subtracting N0/N1

        if resp_err:
            resp_bin, *_, resp_cov = bin_Cls(self.resp2_cls, bins=bins)[1:]
            k = self.y / resp_bin
            resp_cov = np.einsum('ij,i,j->ij', resp_cov / self.N_N1, k, k)
            self.cov_sys += resp_cov

        self.cov = self.cov + self.cov_sys
        self.yerr = np.sqrt(np.diag(self.cov))
        self.yerr_mean = np.sqrt(np.diag(self.cov - self.cov_sys) / self.N + np.diag(self.cov_sys))

    def _add_or_sub(self, other, op):
        new = LensingSpectra.__new__(LensingSpectra)

        for key in ['N', 'N_N1', 'do_data', 'do_SAN0', 'Lmax']:
            assert getattr(self, key) == getattr(other, key)
            setattr(new, key, getattr(self, key))

        new.config = self.config
        new.N0s = op(self.N0s, other.N0s)
        new.Cls = op(self.Cls, other.Cls)
        if self.N_N1 > 0:
            new.N1s = op(self.N1s, other.N1s)
        if self.do_data:
            new.Cl0 = op(self.Cl0, other.Cl0)
        if self.do_SAN0:
            new.SAN0s = op(self.SAN0s, other.SAN0s)
        return new

    def __add__(self, other):
        return self._add_or_sub(other, operator.add)

    def __sub__(self, other):
        return self._add_or_sub(other, operator.sub)
