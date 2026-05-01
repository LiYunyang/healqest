import os
import subprocess
import sys
from typing import Union
import tempfile as tf

import numpy as np
import healpy as hp
from healqest import healqest_utils as hq, log

logger = log.get_logger(__name__)


def kspice(  # noqa: C901
    m1: Union[np.ndarray, str, list],
    m2: Union[np.ndarray, str, list] = None,
    weight1: Union[np.ndarray, str] = None,
    weight2: Union[np.ndarray, str] = None,
    *,
    lmax=-1,
    apodizetype=1,
    apodizesigma: Union[float, str] = "NO",
    thetamax: float = 180,
    tolerance: float = 5e-8,
    subav: bool = False,
    subdipole: bool = False,
    script=False,
    cl_out: str = None,
    spice: str = None,
    kernel=False,
):
    """
    A python wrapper for PolSpice for temperature (kappa) file only.

    Notes
    -----
    For ka

    Parameters
    ----------
    m1: np.ndarray(3, npix)
        map1 for PS estimation.
    m2: np.ndarray(3, npix), optional.
        map2 for cross PS estimation. If None, m2=m1. Default: None.
    weight1, weight2: np.ndarray=None.
        shape (npix,). The weight map for map1/2. If None,no weights are applied.
        Note that when the `weight2` is None, if `m2` is specified, `weight2` will be considered as FULL SKY
        rather than the same as `weight1`.
    lmax: int=-1.
        The maximum ell number for PS computation. It is advised set lmax=3*nside-1
        (or lmax=-1) for minimum aliasing.
    apodizetype: int=1.
        The apodization type for angular correlation function apodization.
            - 0: the correlation function is multiplied by a gaussian window
                + equal to 1 at theta=0.
                + equal to 0.5 at theta= -apodizesigma/2.
                + equal to 1/16 at theta= -apodizesigma.
            - 1: the correlation function is multiplied by a cosine window
                + equal to 1 at theta=0.
                + equal to 0.5 at theta= -apodizesigma/2.
                + equal to 0 at theta= -apodizesigma.
    apodizesigma: float or str='NO'.
        scale factor in DEGREES of the correlation function tappering. For better
        results, ``apodizesigma`` should be close to ``thetamax``. Use 'NO' to
        disable apodization.
    thetamax: float (0-180)=180.
        The maximum angular distance (in deg) for computing angular-correlation
        function.
    tolerance: float=5e-8.
        Tolerance for convergence.
    subav: bool=False.
    subdipole: bool=False.
    script: bool=False
        If True, return the command line script to be executed.
    cl_out: str
        If present, the output Cl will be write to this file
    spice: str=None
        Path to spice binary
    kernel: bool=False
        If True, return the mode coupling matrix of shape (lmax+1, 2lmax+1).

    Returns
    -------
    [command]: str
        The command line script to be executed.
    [clhat]: np.ndarray(1, nlmax+1)
        PS in orders of: TT
    [kernel]: np.ndarray
        shape (lmax+1, 2lmax+1)

    Notes
    -----
    The wrapper forces ``decouple`` to be True.

    References
    ----------
    PolSpice: http://www2.iap.fr/users/hivon/software/PolSpice/README.html
    """
    dtype = np.float64

    # locate spice binary
    if spice is None:
        spice_bin = os.environ.get("POLSPICE_BIN", os.path.expanduser("~/.local/bin"))
        spice = os.path.join(spice_bin, f"spice_SP")
        if not os.path.exists(spice):
            spice = os.path.join(spice_bin, f"spice_DP")
        if not os.path.exists(spice):
            spice = os.path.join(spice_bin, f"spice")
    else:
        assert os.path.exists(spice)

    # locate the cached polspice configuration
    polspice_config = os.path.expanduser("~/.local/share/polspice")
    if not os.path.exists(polspice_config):
        os.makedirs(polspice_config, exist_ok=True)

    command = [
        spice,
        "-verbosity", "0",
        "-nlmax", str(lmax),
        "-overwrite", "YES",
        "-polarization", "NO",
        "-pixelfile", "NO",
        "-pixelfile2", "NO",
        "-decouple", "YES",
        "-symmetric_cl", "NO",
        "-tolerance", str(tolerance),
        "-apodizetype", str(apodizetype),
        "-apodizesigma", str(apodizesigma),
        "-thetamax", str(thetamax),
        "-subav", "NO" if not subav else "YES",
        "-subdipole", "NO" if not subdipole else "YES",
        "-corfile", "NO",
        # "-verbosity", "2",
    ]  # fmt: off
    if m2 is None and weight2 is not None:
        # normally we don't want to do this
        m2 = m1
    with tf.TemporaryDirectory(prefix='spice') as tmp:
        from astropy.io import fits

        for item, name in zip(
            [m1, weight1, m2, weight2], ['mapfile', 'weightfile', 'mapfile2', 'weightfile2']
        ):
            if item is not None:
                if isinstance(item, str):
                    fname = item
                else:
                    fname = os.path.join(tmp, f"{name}.fits")
                    hp.write_map(
                        fname,
                        item,
                        overwrite=True,
                        dtype=dtype,
                        partial=True if name.startswith("mapfile") else False,
                    )
                command += [f"-{name}", fname]
        if cl_out is None:
            cl_out = os.path.join(tmp, f"cls.dat")
        command += [f"-clfile", cl_out]
        if kernel:
            kernel_out = os.path.join(tmp, f"kernel.dat")
            command += ["-kernelsfileout", kernel_out]
        if script:
            return command
        try:
            result = subprocess.run(command, capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed with exit code {e.returncode}")
            print(f"🔧 Command: {' '.join(e.cmd)}")
            if e.stdout:
                print(f"📤 Stdout:\n{e.stdout.strip()}")
            if e.stderr:
                print(f"📣 Stderr:\n{e.stderr.strip()}")
            sys.exit(e.returncode)

        if kernel:
            return fits.open(kernel_out)[0].data[0, :, :].T
        try:
            ell, *clhat = np.loadtxt(cl_out).T
        except ValueError as e:
            print(result.stdout)
            raise e
        clhat = np.array(clhat[0])
        return clhat


def kappa_spectrum(  # noqa: C901
    m1: Union[np.ndarray, str, list],
    m2: Union[np.ndarray, str, list] = None,
    mask1: Union[np.ndarray, str] = None,
    mask2: Union[np.ndarray, str] = None,
    mask_alm=True,
    g=None,
    anafast=True,
    nside=None,
    cl_out: str = None,
    **kwargs,
):
    """
    General power spectrum estimator.

    Parameters
    ----------
    m1, m2: np.ndarray or str
        1d array of map or file(.fits) name of maps. If `synfast=True`, then m1/m2 can be alm/map array or map
        fname, but if `synfast=False`, then m1/m2 should be map fnames or map arrays.
    mask1, mask2: np.ndarray or str
        mask (binary or float) array or file names.
    mask_alm: bool=True
        If true, assume the input alm is unmasked and apply alm2map->mask->map2alm operations.
    g: Geometry
        ducc wrapper object. `g.nside` attribute should match that of the maps.
    anafast: bool=True
        If True, use hp.alm2cl to perform quick power spectrum estimation. The fsky-correction is
        automatically applied so the returned power spectrum should be unbiased.
    nside: int
        used to convert alm2map if `m1/m2` are alm objects and `g` is not given. This is ignore if
        `anafast=True`.
    cl_out: str=None
        Optional output directory.
    kwargs: dict
        kspice keyword arguments.
    """

    def _alm2alm(obj, mask_obj):
        if isinstance(mask_obj, str):
            mask = hp.read_map(mask_obj)
        else:
            mask = mask_obj
        lmax = None
        if isinstance(obj, str):
            m = hp.read_map(obj)
        else:
            if hq.map_or_alm(obj):
                m = obj
            else:
                lmax = hp.Alm.getlmax(len(obj))
                if mask is not None and mask_alm:
                    if g is None:
                        nside = hp.get_nside(mask)
                        m = hp.alm2map(obj, nside=nside)
                    else:
                        m = g.alm2map(obj)
                else:
                    return obj, mask
        if m is not None:
            if mask is not None:
                m *= mask
            func = hp.map2alm if g is None else g.map2alm
            return func(m, iter=0, lmax=lmax), mask
        raise ValueError

    if anafast:
        alm1, mask1 = _alm2alm(m1, mask1)
        if m2 is None:
            out = hp.alm2cl(alm1)
        else:
            if mask2 is None:
                mask2 = mask1
            alm2, mask2 = _alm2alm(m2, mask2)
            out = hp.alm2cl(alm1, alm2)
        if mask1 is None:
            fsky = 1
        elif mask2 is None:
            fsky = np.mean(mask1)
        else:
            fsky = np.mean(mask1 * mask2)
        out /= fsky
        if cl_out is not None:
            l = np.arange(out.shape[-1])
            np.savetxt(cl_out, np.array([l, out]).T)
        return out
    else:
        data = {'m1': m1, 'm2': m2}
        for key, obj in data.items():
            if obj is not None and not isinstance(obj, str):
                if not hq.map_or_alm(obj):
                    if g is None:
                        data[key] = hp.alm2map(obj, nside=nside)
                    else:
                        data[key] = g.alm2map(obj)
                else:
                    data[key] = np.asarray(obj, dtype=np.float64)
        # if m1/m2 are given as file names, then they are assumed to be maps.
        return kspice(m1=data['m1'], m2=data['m2'], weight1=mask1, weight2=mask2, cl_out=cl_out, **kwargs)


def get_spice_kernel(nside, lmax, thetamax=None, apodizesigma=None, apodizetype=None):
    """Return the polspice coupling kernel of shape (lmax+1, 2lmax+1)."""
    if thetamax is None:
        thetamax = kspice.__kwdefaults__['thetamax']
    if apodizesigma is None:
        apodizesigma = kspice.__kwdefaults__['apodizesigma']
    if apodizetype is None:
        apodizetype = kspice.__kwdefaults__['apodizetype']
    fname = f"n{nside}_lmax{lmax}_thetamax{thetamax}_apodizesigma{apodizesigma}_apodizetype{apodizetype}.npy"
    cache_dir = os.environ.get("HEALQEST_IO_ROOT")
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~/.local/share"), "healqest")
    else:
        cache_dir = os.path.join(cache_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    path = os.path.join(cache_dir, fname)
    if os.path.exists(path):
        return np.load(path)
    else:
        logger.warning(f"cache {fname} not found, computing it...")
        zero = np.random.normal(0, 1, hp.nside2npix(nside))
        K = kspice(
            m1=zero,
            lmax=lmax,
            thetamax=thetamax,
            apodizesigma=apodizesigma,
            apodizetype=apodizetype,
            kernel=True,
        )
        logger.info(f"cache saved to {path}")
        np.save(path, K)
        return K


class KappaMap:
    """
    Kappa (convergence) map container for power spectrum estimation.

    Wraps either a mean-field-subtracted QE reconstruction or an input kappa map used for
    cross-correlations. Maps can be kept in-memory or written to disk for PolSpice.
    """

    def __init__(self, config, i, ktype, mvtype=None, mf_group=None, N1=None, cmbset=None, outdir=None,
                 curl=False):  # fmt: off
        """
        Kappa (convergence) map container for power spectrum estimation.

        Parameters
        ----------
        config: Config
            Pipeline configuration object.
        i: int
            Simulation seed index.
        ktype: str or None
            Reconstruction type (e.g. ``'xx'``, ``'xy'``). Falsy values mark this map as an input
            kappa map for cross-correlation rather than a QE reconstruction.
        mvtype: str, optional
            Minimum-variance estimator tag.
        mf_group: int, optional
            Mean-field group index: 0 = all sims, 1 = first half, 2 = second half.
        N1: bool, optional
            If True, use the N1-bias simulation file range.
        cmbset: str, optional
            CMB dataset identifier (e.g. ``'a'``).
        outdir: str, optional
            Directory to write FITS maps to disk (required for PolSpice).
        curl: bool, optional
            If True, use the curl (B-mode) estimator instead of the gradient estimator.
        """
        self.config = config
        self.i = i
        if not ktype:
            # this is the input map to cross-with
            self.cross = True
            ktype = None
            mvtype = None
            N1 = None
            cmbset = None
        else:
            self.cross = False
        self.ktype = ktype
        self.mvtype = mvtype
        self.mf_group = mf_group
        self.N1 = N1
        self.cmbset = cmbset
        self.outdir = outdir
        self.curl = curl
        if config.nbundle is not None:
            self.bundle_loop = config.bundle_pairs
            self.bundle_keys = [f'X{b}' for b in range(self.config.nbundle)] + ['X']
        else:
            self.bundle_keys = [None]
            self.bundle_loop = [None]
        if self.cross:
            self._make_map_kin()  # input kappa map from alms. Optionally save to disk for polspice.
        else:
            self._make_map()  # create the mf-subtracted kappa maps. Optionally save to disk for polspice.
        self.file_mask = self.config.tmp_file_mask  # the on-disk mask file for polspice.

    def _save_map(self, kmap, bundle_key=None):
        fname = self.get_fname(bundle_key=bundle_key)
        assert kmap.ndim == 1
        kmap[kmap == 0] = hp.UNSEEN  # important for partial maps.
        hp.write_map(fname, kmap, overwrite=True, dtype=np.float64, partial=True)  # polspice needs float64

    def _get_mf(self, y=None, mf_group=0, bundle=None):
        """
        Load and normalize the stacked mean-field map for a given bundle.

        Parameters
        ----------
        y: np.ndarray, optional
            Raw QE map for this simulation, used for leave-one-out normalization.
        mf_group: int, optional
            Mean-field group: 0 = all sims, 1 = first half, 2 = second half.
        bundle: tuple or None, optional
            Bundle pair identifier passed through to the file path resolver.

        Returns
        -------
        np.ndarray
            Normalized mean-field map (same pixelization as the QE map).
        """
        assert mf_group in [0, 1, 2]
        file_mf = self.config.p_plm(
            tag=self.mvtype, stack_type=self.ktype, N1=self.N1, bundle=bundle, cmbset=self.cmbset
        )
        logger.warning(f"using MF: {os.path.basename(file_mf)}")
        i1, i2 = self.config.sim_range_N1 if self.N1 else self.config.sim_range

        # read maps and bad pixels are set to 0
        gctag = 'gmf' if not self.curl else 'cmf'
        field = gctag if mf_group == 0 else f"{gctag}{mf_group}"
        mf, h = hq.read_map(file_mf, h=True, field=field, dtype=np.float64, return_cosmo=False)
        h = dict(h)
        assert h['NSIM'] == (i2 - i1 + 1), f"loaded MF ({h['NSIM']}) is inconsistent with config settings!"
        nsim = h[f'NSIM'] if mf_group == 0 else h[f'NSIM{mf_group}']
        split_i = h['SPLITIDX']

        if self.i == 0:
            assert self.ktype == 'xx'
            mf = mf / nsim
        else:
            if (
                (mf_group == 1 and self.i < split_i + 1)
                or (mf_group == 2 and self.i >= split_i + 1)
                or mf_group == 0
            ):
                mf = (mf - y) / (nsim - 1)
            else:
                mf = mf / nsim
        return mf

    def _make_map(self):
        self.kmaps = {k: 0 for k in self.bundle_keys}
        for bundle_pair in self.bundle_loop:
            file_plm = self.config.p_plm(
                tag=self.mvtype,
                seed1=self.i,
                ktype=self.ktype,
                cmbset=self.cmbset,
                N1=self.N1,
                bundle=bundle_pair,
            )
            y = hq.read_map(file_plm, field=0 if not self.curl else 1, dtype=np.float64, return_cosmo=False)
            mf = self._get_mf(mf_group=self.mf_group, y=y, bundle=bundle_pair)
            kmv = y - mf

            if self.config.nbundle is not None:
                for b in bundle_pair:
                    self.kmaps[f'X{b}'] += kmv
                    # should divide by `config.nbundle`, but this is taken care of in `compute_ps`
                self.kmaps['X'] += kmv
                # should divide by `config.nbundle^2/2`, but this is taken care of in `compute_ps`
            else:
                self.kmaps[bundle_pair] = kmv

        if self.outdir:
            for key, kmv in self.kmaps.items():
                self._save_map(kmv, bundle_key=key)

    def _make_map_kin(self):
        self.kmaps = {None: None}
        if os.path.exists(self.get_fname()) and self.outdir is not None:
            pass
        else:
            fname = self.config.path(self.config.kappa_in, seed=self.i)
            ilm = hp.read_alm(fname)
            ilm = hq.reduce_lmax(ilm, lmax=self.config.Lmax)
            kmv = self.config.g.alm2map(np.nan_to_num(ilm))
            if self.outdir:
                self._save_map(kmv)
            else:
                self.kmaps[None] = kmv

    def get_fname(self, bundle_key=None):
        if self.cross:
            fname = f"kappa_in_{self.i:03d}.fits"
        else:
            s1, s2, cmbset1, cmbset2 = self.config.ktype2ij(self.ktype, self.i, j=None, cmbset=self.cmbset)
            fname = self.config.f_tmp(
                tag=self.mvtype,
                seed1=s1,
                seed2=s2,
                ktype=self.ktype,
                N1=self.N1,
                mf_group=self.mf_group,
                bundle=bundle_key,
                cmbset1=cmbset1,
                cmbset2=cmbset2,
                curl=self.curl,
            )
        return os.path.join(self.outdir, fname)

    def mask_bias(self, obj):
        """Compute the spectrum bias due to masking from two kappa map objects."""
        # HACK: this is a crude approximation to mask^2, because it depends on the QE type.
        n_qe = np.count_nonzero([o.cross is False for o in [obj, self]])
        assert n_qe in [1, 2], "either one or both of the maps should be QE maps (not input kappa maps)"
        fsky_qe2 = self.config.mask_cinv['t'] * self.config.mask_cinv['p']
        return np.mean(fsky_qe2**n_qe * self.config.mask_ps**2) / np.mean(self.config.mask_ps**2)


def compute_ps_single(mobj1, mobj2, bundle1, bundle2):
    m1 = mobj1.get_fname(bundle_key=bundle1)
    m2 = mobj2.get_fname(bundle_key=bundle2)
    clkk = kappa_spectrum(m1, m2, mask1=mobj1.file_mask,mask2=mobj2.file_mask, g=mobj1.config.g,
                          anafast=False, **mobj1.config.spice_kwargs)  # fmt: off
    return clkk


def compute_ps(mobj1: KappaMap, mobj2: KappaMap, save=False, skip=False):
    """
    Compute power spectrum of two kappa map objects.

    Parameters
    ----------
    mobj1, mobj2: KappaMap
        The kappa map objects to cross-correlate. If `mobj2.ktype` is None, it will be treated as the input
        kappa map.
    save: bool=False
        If True, save the output power spectrum to disk.
    skip: bool=False:
        If True, skip the computation if the output file already exists.

    """
    if not mobj2.cross:
        assert mobj1.cmbset == mobj2.cmbset
        assert mobj1.curl == mobj2.curl
        assert mobj1.N1 == mobj2.N1
    assert mobj1.i == mobj2.i
    name = f"{mobj1.mvtype}"
    k1, k2 = mobj1.ktype, mobj2.ktype

    if save:
        cl_out = mobj1.config.p_cls(tag=name, seed1=mobj1.i, seed2=None, ktype1=k1, ktype2=k2,
                                    N1=mobj1.N1, ext='dat', cmbset=mobj1.cmbset, curl=mobj1.curl)  # fmt: off
        if skip and os.path.exists(cl_out):
            logger.warning(f"Skipping {cl_out}", extra={"force": True})
            return None
        os.makedirs(os.path.dirname(cl_out), exist_ok=True)

    out = []
    if mobj1.config.nbundle is not None:
        m = mobj1.config.nbundle
        clx = compute_ps_single(mobj1, mobj2, bundle1='X', bundle2='X')
        clxj = 0  # second term in Eq 38
        clj = 0  # third term in Eq 38
        out = []
        for j in range(m):
            cl = compute_ps_single(mobj1, mobj2, bundle1=f'X{j}', bundle2=f'X{j}')
            clxj += cl
            out.append(cl)
        for bp in mobj1.config.bundle_pairs:
            cl = compute_ps_single(mobj1, mobj2, bundle1=bp, bundle2=bp)
            clj += cl
            out.append(cl)
        cl_tot = 4 / (m * (m - 1) * (m - 2) * (m - 3)) * (clx - clxj + clj)
        out.append(cl_tot)
    else:
        clkk = compute_ps_single(mobj1, mobj2, bundle1=None, bundle2=None)
        out.append(clkk)

    mask_bias = mobj1.mask_bias(mobj2)
    out = np.array(out) / mask_bias
    if save:
        header = f"# nlmax, ncor, nside = {mobj1.config.Lmax:8d} {1:8d} {mobj1.config.g.nside:8d}"
        hq.write_cl(cl_out, out, header=header)
    return out
