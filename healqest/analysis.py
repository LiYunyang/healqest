import re
import numpy as np
import operator
from healqest import log

logger = log.get_logger(__name__)


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


def unbin_spectrum(Clb, bins, lmax):
    assert Clb.shape[0] == len(bins) - 1, "the number of binned Cls should match the number of bins"
    ell = np.arange(lmax + 1)
    bin_idx = np.digitize(ell, bins, right=False)
    out = np.zeros(lmax + 1)
    sel = np.logical_and(bin_idx <= len(bins) - 1, bin_idx > 0)
    out[sel] = Clb[bin_idx[sel] - 1]
    return out


def bin_Cls(Cls, bins, return_ensemble=False):
    """Return bin center, bined Cls, error on the mean and cov.

    Parameters
    ----------
    Cls: np.ndarray
        shape (nspec, nell) or (nell, )
    bins: np.ndarray
        shape (nbin+1, )
    return_ensemble: bool=False
        If True, return the binned Cls for each realization, otherwise return the mean.
    """
    x = (bins[1:] + bins[:-1]) / 2
    Cbs = np.array([bin_spectrum(_, bins=bins, verbose=False)[1] for _ in np.atleast_2d(Cls)])
    cov = np.cov(Cbs, rowvar=False)
    return x, Cbs if return_ensemble else np.mean(Cbs, axis=0), cov


def load_sql(seeds, config, spec_type, mvtype, curl, ops: str, Lmax=None, **kw):
    """
    Load spectra from the sqlite database and performs specific coadding procedure.

    Parameters
    ----------
    seeds: list of int
        The seeds of the spectra to be loaded.
    config: Config
    spec_type: str
        The type of the spectra to be loaded, e.g., 'n0', 'n1', 'san0', 'rdn0'.
    mvtype: str
        The MV type of the spectra to be loaded, e.g., 'TT', 'EB'.
    curl: bool
        Whether to load the curl mode spectra.
    ops: str
        Instructions on how the spectra should be combined. The instruction will be parsed to be signs and
        spec types. For example, "xyxy-xyyx" means loading the 'xyxy' and 'xyyx' spectra and coadding them
        with + and - signs respectively.
    Lmax: int=None

    Returns
    -------
    np.ndarray
        The coadded spectra.
    """
    db = config.get_sql_table(mvtype, spec_type=spec_type, curl=curl)
    operators = re.split(r'([+-])', ops)
    if operators[0] not in ['+', '-']:
        operators = ['+'] + operators

    assert len(operators) % 2 == 0, "invalid ops format"
    out = list()
    with db:
        for i in seeds:
            cl = 0
            for s, ktype in zip(operators[0::2], operators[1::2]):
                k1 = ktype[:2]
                k2 = ktype[2:] or None
                _db, sql_key = config.get_sql_keys(seed=i, ktype1=k1, ktype2=k2, **kw, curl=curl, tag=mvtype)
                assert _db == db, (str(_db), str(db), kw)
                sign = -1 if s == '-' else 1
                _cl = db.query_conn(sql_key)
                if _cl is None:
                    raise ValueError(f"Failed to load {sql_key} from {db}")
                cl += sign * _cl
            out.append(cl)
    out = np.array(out)
    if Lmax is not None:
        out = out[:, : Lmax + 1]
    return out


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
        calibrate_SAN0=True,
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
            If True, average all spectra in the file, otherwise return the last column.
        coadd: bool
            Special case to load spectrum from `cls_coadd/` instead of `cls/`, where the lensing
            reconstruction map is coadded before taking spectra.
        do_SAN0: bool
            If True, load SAN0 spectra and subtract it from the sims spectra.
        do_RDN0: bool
            If True, load RDN0 spectra and subtract it from the data spectrum.
        do_data: bool
            If True, load the data spectrum and subtract N0 and N1 from it.
        calibrate_SAN0: bool
            If True, calibrate the SAN0 spectra by N0.
        """
        self.config = config
        self.cmbset = cmbset
        self.curl = curl
        self.calibrate_SAN0 = calibrate_SAN0
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
        self.ys = None  # binned sims spectrum for each realization
        self.cov = None  # total covariance
        self.cov_sys = None  # systematic part of the covariance
        self.cov_sm = None  # covariance cor sim-mean
        self.yerr = None  # single rlz error, including sys and stats parts
        self.yerr_mean = None  # error on the mean. Only the stats part is reduced by sqrt(N).

    @staticmethod
    def smooth_resp(y, seq):
        from scipy.signal import savgol_filter

        out = y.copy()
        for s in seq:
            out[s:] = savgol_filter(y, window_length=s, polyorder=3)[s:]
        return out

    @property
    def fsky(self):
        return np.mean(self.config.mask_ps**2)

    @property
    def clkk(self):
        return self.config.clkk

    @property
    def N0(self):
        return np.mean(self.N0s, axis=0)

    @property
    def N1(self):
        if self.N1s is not None:
            return np.mean(self.N1s, axis=0)
        else:
            return np.zeros(self.Lmax + 1)

    def load_resp(self, resp_smooth=None):  # noqa: C901
        if self.resp_type == 'auto':
            if self.N_N1 > 0 and not self.curl:
                Cls_ab = load_sql(
                    range(1, self.N_N1 + 1),
                    config=self.config,
                    spec_type='n1',
                    mvtype=self.mvtype,
                    curl=self.curl,
                    ops='aabb',
                    N1=True,
                    Lmax=self.Lmax,
                )
                self.resp2_cls = Cls_ab / self.clkk[: self.Lmax + 1]
                self.resp2 = np.mean(Cls_ab / self.clkk[: self.Lmax + 1], axis=0)
            else:
                if self.N_N1 > 0 and self.curl:
                    logger.warning("Ignoring resp for curl mode.")
                logger.info("not loading resp function due to no N1 sims")
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
            logger.warning("disable resp function")
            self.resp2 = np.ones(self.Lmax + 1)
            return
        if resp_smooth is not None:
            self.resp2 = self.smooth_resp(self.resp2, np.atleast_1d(resp_smooth))

    def offload(self):
        bad_k = [k for k, v in self.config.__dict__.items() if k.startswith('mask')]
        for k in bad_k:
            del self.config.__dict__[k]

    @property
    def hartlap(self):
        """Divide the inverse covariance by this factor, because the invcov is overestimated."""
        return self.N / (self.N - self.cov.shape[0] - 1)

    @property
    def snr(self):
        snr = np.sqrt(np.sum(np.linalg.inv(self.cov) / self.hartlap))
        return snr

    def load(self):
        self.offload()

        kw = dict(Lmax=self.Lmax, curl=self.curl, mvtype=self.mvtype, config=self.config)
        N0_loop = range(1, self.N + 1)
        N1_loop = range(1, self.N_N1 + 1) if self.N_N1 > 0 else []

        N0s = load_sql(N0_loop, spec_type='n0', ops='xyxy+xyyx', cmbset=self.cmbset, **kw)
        Cls_hat = load_sql(N0_loop, spec_type='n0', ops='xxxx', cmbset=self.cmbset, **kw)

        self.N0s = N0s[:, : self.Lmax + 1] / self.resp2

        if self.N_N1 > 0:
            N1s = load_sql(N1_loop, spec_type='n1', ops='abab+abba-xyxy-xyyx', cmbset='a', N1=True, **kw)
            self.N1s = N1s[:, : self.Lmax + 1] / self.resp2
        else:
            self.N1s = None

        self.Cls_hat = Cls_hat[:, : self.Lmax + 1] / self.resp2
        if self.do_RDN0:
            RDN0s = load_sql(N0_loop, spec_type='rdn0', ops='x0x0+x00x+0xx0+0x0x', cmbset=self.cmbset, **kw)
            RDN0s -= N0s
            self.RDN0 = np.mean(RDN0s[:, : self.Lmax + 1] / self.resp2, axis=0)

        if self.do_data:
            Cl0 = load_sql([0], spec_type='rdn0', ops='xxxx', cmbset=self.cmbset, **kw)[0]
            self.Cl0 = Cl0[: self.Lmax + 1] / self.resp2
            if self.do_RDN0:
                self.Cl0 -= self.RDN0 + self.N1
            else:
                self.Cl0 -= self.N0 + self.N1

        if self.do_SAN0:
            SAN0s = load_sql(N0_loop, spec_type='san0', ops='xxxx', cmbset=self.cmbset, SAN0=True, **kw)
            self.SAN0s = SAN0s[:, : self.Lmax + 1] / self.resp2
            if self.calibrate_SAN0:
                logger.info("calibrate SAN0 by N0")
                self.SAN0s *= self.N0 / np.mean(self.SAN0s, axis=0)
            self.Cls = self.Cls_hat - self.SAN0s - self.N1
        else:
            self.Cls = self.Cls_hat - self.N0 - self.N1

    def bin_spec(self, bins, norm_cl=None, resp_err=False):
        if norm_cl is not None:
            norm_cb = bin_spectrum(norm_cl[: self.Lmax + 1], bins=bins, verbose=False)[1]
            fac = 1 / unbin_spectrum(norm_cb, bins=bins, lmax=self.Lmax)
            # fac = 1 / norm_cl[: self.Lmax + 1]
        else:
            fac = 1

        self.x = (bins[1:] + bins[:-1]) / 2
        N0cov = bin_Cls(self.N0s * fac, bins)[-1]
        if self.N_N1 > 0:
            N1cov = bin_Cls(self.N1s * fac, bins)[-1]
        else:
            N1cov = np.diag(np.zeros(len(self.x)))

        if self.do_data:
            self.y0 = bin_Cls(self.Cl0 * fac, bins=bins)[1]

        self.ys, self.cov = bin_Cls(self.Cls * fac, bins=bins, return_ensemble=True)[1:]
        self.y = np.mean(self.ys, axis=0)

        self.cov_sys = np.zeros_like(self.cov)
        # because of SAN0 normalization, the sim-mean cov is always from Cls-N0s
        self.cov_sm = bin_Cls((self.Cls_hat - self.N0s) * fac, bins=bins)[-1] / self.N

        if not self.do_SAN0:
            logger.warning("cov might be overestimated because SAN0 is not subtracted")
            self.cov_sys += N0cov / self.N  # systematic err from subtracting N0

        if self.N_N1 > 0:
            self.cov_sys += N1cov / self.N_N1  # systematic err from subtracting N1
            self.cov_sm += N1cov / self.N_N1

        if resp_err and self.N_N1 > 0:
            # the response corrected spectra is y = s/<r>, and the total variance is:
            # var(y) = var(s)/<r>^2 + var(<r>)*s^2/<r>^4
            # the first term is the usual covariance, and the second term is:
            # var(<r>)*s^2/<r>^4 = var(r)/N * (y/<r>)^2
            resp_bin, resp_cov = bin_Cls(self.resp2_cls, bins=bins)[1:]
            k = self.y / resp_bin
            resp_cov = np.einsum('ij,i,j->ij', resp_cov / self.N_N1, k, k)
            self.cov_sys += resp_cov
            self.cov_sm += resp_cov

        self.cov = self.cov + self.cov_sys
        self.yerr = np.sqrt(np.diag(self.cov))
        self.yerr_mean = np.sqrt(np.diag(self.cov_sm))

    def _add_or_sub(self, other, op):
        new = LensingSpectra.__new__(LensingSpectra)

        for key in ['N', 'N_N1', 'do_data', 'do_SAN0', 'Lmax']:
            assert getattr(self, key) == getattr(other, key)
            setattr(new, key, getattr(self, key))

        new.config = self.config
        new.Cls_hat = op(self.Cls_hat, other.Cls_hat)
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
