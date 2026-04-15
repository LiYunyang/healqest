import os
import sys
from astropy.io import fits
import numpy as np
import healpy as hp
import importlib
from pathlib import Path
import yaml
import logging as lg
from tqdm import tqdm
import tempfile as tf
from typing import Union
import subprocess
import hashlib
import string

logger = lg.getLogger(__name__)

np.seterr(divide="ignore", invalid="ignore")


class EvalFormatter(string.Formatter):
    """
    More capable string formatter that can evaluate expressions.

    Usage:
    >>> template_string = "original_{A}_and_lower_{B.lower()}"
    >>> EvalFormatter().format(template_string, A='A', B='B')
    """

    def get_field(self, field_name, args, kwargs):
        try:
            val = eval(field_name, {}, kwargs)
        except (NameError, AttributeError):
            val = super().get_field(field_name, args, kwargs)
        return val, field_name


def reduce_lmax(alm, lmax=4000):
    """Reduce the lmax of input alm."""
    lmaxin = hp.Alm.getlmax(alm.shape[-1])
    logger.debug(f"Reducing lmax: lmax_in={lmaxin} -> lmax_out={lmax}")
    almout = np.zeros((*alm.shape[:-1], hp.Alm.getsize(lmax)), dtype=alm.dtype)
    oldi = 0
    newi = 0
    dl = lmaxin - lmax
    for i in range(0, lmax + 1):
        oldf = oldi + lmaxin + 1 - i
        newf = newi + lmax + 1 - i
        almout[..., newi:newf] = alm[..., oldi : oldf - dl]
        oldi = oldf
        newi = newf
    return almout


def get_nside(lmax):
    """Calculates the most appropriate nside based on lmax."""
    nside = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    idx = np.argmin(np.abs(nside - lmax))
    return nside[idx]


def get_lensedcls(file, lmax=2000, return_dict=False):
    logger.info(f"Loading lensed Cls from {file}")
    ell, *dls = np.loadtxt(file, unpack=True)
    fac = ell * (ell + 1) / 2 / np.pi
    cls = dls / fac
    ell = np.concatenate([[0, 1], ell])[: lmax + 1]
    cls = np.pad(cls, ((0, 0), (2, 0)), mode='constant', constant_values=0)[..., : lmax + 1]
    if not return_dict:
        return ell, *cls
    else:
        return {key: cls[i] for i, key in enumerate(['tt', 'ee', 'bb', 'te'])}


def get_unlensedcls(file, lmax=2000):
    logger.info(f"Loading unlensed Cls from {file}")
    # order: TT/EE/BB/TE/PP/TP/EP
    ell, *dls = np.loadtxt(file, unpack=True)
    fac = ell * (ell + 1) / 2 / np.pi
    fac_p = np.sqrt(ell * (ell + 1))
    cls = dls / fac
    cls[4] /= fac_p**2
    cls[5:] /= fac_p

    ell = np.concatenate([[0, 1], ell])[: lmax + 1]
    cls = np.pad(cls, ((0, 0), (2, 0)), mode='constant', constant_values=0)[..., : lmax + 1]
    return ell, *cls


def get_qes(qeset):
    """Retrieve the estimators needed to compute a given QE set.

    Parameters
    ----------
    qeset : str
        A string representing the QE set.

    Returns
    -------
    list or None
        A list of estimators neesed.
    """
    single = {
        "TT",
        "EE",
        "TE",
        "EB",
        "TB",
        "ET",
        "BE",
        "BT",
        "TTbhTTprf",
        "GMVbhTTprf",
        "GMVTTEETEbhTTprf",
        'TTprf',
    }

    composite = {
        "GMV": ["TT", "EE", "EB", "TE", "TB", "EB", "TE", "TB"],
        "GMVTTEETE": ["TT", "EE", "TE", "ET"],
        "GMVTBEB": ["TB", "BT", "EB", "BE"],
        "MV": ["TT", "EE", "EB", "TE", "TB", "BE", "ET", "BT"],
        "qMV": ["TT", "EE", "EB", "TE", "TB"],
        "MVnoTT": ["EE", "EB", "TE", "TB", "EB", "TE", "TB"],
        "qMVnoTT": ["EE", "EB", "TE", "TB"],
        "MVnoEB": ["EE", "TE", "TB", "TE", "TB"],
        "qMVnoEB": ["EE", "TE", "TB"],
        "TTEETE": ["TT", "EE", "TE", "ET"],
        "qTTEETE": ["TT", "EE", "TE"],
        "TBEB": ["TB", "BT", "EB", "BE"],
        "qTBEB": ["TB", "EB"],
        "PP": ["EE", "EB", "BE"],
        "qPP": ["EE", "EB"],
        "TEET": ["TE", "ET"],
        "EBBE": ["EB", "BE"],
        "TBBT": ["TB", "BT"],
        "qTEET": ["TE"],
        "qEBBE": ["EB"],
        "qTBBT": ["TB"],
        "qMVTTbhTTprf": ["TTbhTTprf", "EE", "TE", "TB", "EB"],
        "MVTTbhTTprf": ["TTbhTTprf", "EE", "EB", "TE", "TB", "EB", "TE", "TB"],
    }

    if qeset in composite:
        return composite[qeset]

    # For any qetype that is one of the single entry codes.
    elif qeset in single:
        return [qeset]

    else:
        raise ValueError(f"Unknown QE set: {qeset}")


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


def map_or_alm(m):
    """Check if object is a map (True) or an alm object."""
    try:
        nside = hp.get_nside(m)
        return True
    except TypeError:
        return False


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
            if map_or_alm(obj):
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
                if not map_or_alm(obj):
                    if g is None:
                        data[key] = hp.alm2map(obj, nside=nside)
                    else:
                        data[key] = g.alm2map(obj)
                else:
                    data[key] = np.asarray(obj, dtype=np.float64)
        # if m1/m2 are given as file names, then they are assumed to be maps.
        return kspice(m1=data['m1'], m2=data['m2'], weight1=mask1, weight2=mask2, cl_out=cl_out, **kwargs)


def find_index_file(fname, max_depth=3):
    idx_dir = fname
    c = max_depth
    while c > 0:
        idx_dir = os.path.dirname(idx_dir)
        try:
            loaded = np.load(os.path.join(idx_dir, 'index.npz'))
            return loaded
        except FileNotFoundError:
            c -= 1
    else:
        raise FileNotFoundError(f'partial map index file not found recursively under {idx_dir}')


def read_map(fname, field=(0,), dtype=None, hdu=1, h=False, return_cosmo=True):
    """A wrapper to read the partial maps, as fits or npy files.

    Parameters
    ----------
    fname: str
        a path to '.npy' or '.fits' file.
    field: int/str or list of int/str
        column(s) to read from the FITS file or column index for npy arrays. If now, grab all data columns.
    dtype: str or type
    hdu : int, optional
        the header number to look at (start at 0)
    h : bool, optional
        If True, return also the header. Default: False.
    return_cosmo: bool=False
        If True, make sure the return polarization map is in COSMOS format (flipping U sign if needed).
    """
    if isinstance(field, (str, int)):
        field = [field]

    def _allocate(nside):
        return np.zeros((len(field), hp.nside2npix(nside)), dtype=dtype)

    try:
        if os.path.splitext(fname)[1] == '.npy':
            """load npy partial maps with index stored in parent directories"""
            index_file = find_index_file(fname)
            index = index_file['index']
            m = np.load(fname, mmap_mode='r')
            if field is None:
                field = np.arange(m.shape[0])
            out = _allocate(nside=index_file['nside'])

            for idx, j in enumerate(field):
                out[idx, index] = m[j]
            return np.squeeze(out)
        else:
            """load fits partial maps"""
            return read_map_fits(fname, field=field, dtype=dtype, hdu=hdu, h=h, return_cosmo=return_cosmo)
    except Exception as e:
        raise e from IOError(f"Error reading file: {fname}")


def read_map_fits(fname, field=(0,), dtype=float, hdu=1, h=False, return_cosmo=True):  # noqa: C901
    """Read healpy maps.

    Parameters
    ----------
    fname: str
        a path to '.npy' or '.fits' file.
    field: int/str or list of int/str
        column(s) to read from the FITS file or column index for npy arrays. If now, grab all data columns.
    dtype: str or type=None
    hdu : int, optional
        the header number to look at (start at 0)
    h : bool, optional
        If True, return also the header. Default: False.
    return_cosmo: bool=True
        If True, make sure the return polarization map is in COSMO format (flipping U sign if needed).
    """
    from astropy.io import fits

    if isinstance(field, (str, int)):
        field = [field]

    def _allocate(nside):
        return np.zeros((len(field), hp.nside2npix(nside)), dtype=dtype)

    basename = os.path.basename(fname)
    with fits.open(fname, memmap=True) as hdul:
        names = hdul[hdu].columns.names.copy()
        try:
            # for partial maps, we skip the index column
            names.remove('PIXEL')
        except ValueError:
            pass
        fields_num = []
        fields_name = []
        if field is None:
            field = names
        for c in field:
            if isinstance(c, str):
                if c in names:
                    fields_num.append(names.index(c))
                    fields_name.append(c)
                else:
                    raise ValueError(f"Column {c} not found in the FITS file: {names}")
            elif isinstance(c, (int, np.integer)):
                fields_num.append(c)
                fields_name.append(names[c])
            else:
                raise TypeError(f"field {c} ({type(c)})?")

        nside = int(hdul[hdu].header.get("NSIDE"))
        ordering = hdul[hdu].header.get("ORDERING", "UNDEF").strip().lower()
        if ordering == "undef":
            ordering = 'ring'
            logger.info(f"ORDERING undefined, assume RING: {basename}")
        assert ordering in ["nested", "ring"]

        if ordering in ["nested"]:
            logger.info(f"NESTED->RING: {basename}")

        iau2cosmo = False
        polconv = hdul[hdu].header.get('POLCCONV', "UNDEF").strip().lower()

        # only throw warning if u maps are returned.
        return_u = 2 in fields_name or 'U_POLARISATION' in fields_name

        if return_cosmo:
            if polconv in ['healpix', "cosmo"]:
                pass
            elif polconv in ['iau']:
                iau2cosmo = True
            elif polconv == "undef":
                polconv = 'cosmo'
                if return_u:
                    logger.info(f"POLCONV undefined, assume COSMO: {basename}")
            else:
                if return_u:
                    logger.warning(f"Unrecogonized POLCONV={polconv}, assume COSMO: {basename}")

        out = _allocate(nside=nside)
        partial = 'PIXEL' in hdul[hdu].columns.names
        for j, name in enumerate(fields_name):
            if iau2cosmo and (fields_num[j] == 2 or name == 'U_POLARISATION'):
                logger.warning(f"{polconv} to COSMO: {fields_num[j]}th map {name} of {basename}")
                fac = -1
            else:
                fac = 1
            if partial:
                out[j, hdul[hdu].data['PIXEL']] = hdul[hdu].data[name]
            else:
                out[j, :] = hdul[hdu].data.field(name).astype(dtype, copy=False).ravel()
            out[j] *= fac
            if ordering == "nested":
                idx = hp.ring2nest(nside, np.arange(out[j].size, dtype=np.int32))
                out[j] = out[j][idx]
        if h:
            return np.squeeze(out), hdul[hdu].header
        else:
            return np.squeeze(out)


def generate_seed(seed, cmbset, bundle=None, extra_tag=None):
    """Generate random seed."""
    return int(hashlib.sha256(f"{cmbset}/{seed}/{bundle}/{extra_tag}".encode()).hexdigest()[:8], base=16)


def cinv_io(fname, maps=None, fl=None, eps=None, return_eps=False):
    """
    Read and write cinv maps.

    Parameters
    ----------
    fname : str
        File name.
    maps: array=None
        shape (1, npix) or (3, npix) map. If None, read and return the maps.
    fl: array=None
        shape (1, lmax+1) or (3, lmax+1) for QE weights flT, flE, flB. If None, read and return the weights.
    eps: array= None
        1d array containing the convergence chain of the cinv run.
    return_eps: bool=False
        If True, return the eps chain of the cinv file

    Returns
    -------
    maps: array
        shape (1, npix) or (3, npix) map
    fl: array
        shape (1, lmax+1) or (4, lmax+1) for QE weights flT, flE, flB, flTE
    """
    if maps is None:
        if return_eps:
            hdu = fits.open(fname)[3]
            return hdu.data['eps']
        maps = read_map(fname, field=None, hdu=1, dtype=np.float64)
        hdu = fits.open(fname)[2]
        fl = np.array([hdu.data[_.name] for _ in hdu.columns])
        return np.atleast_2d(maps), fl
    else:
        assert len(maps) in (1, 3)
        assert len(fl) in (1, 4)
        hp.write_map(
            fname, maps, overwrite=True, dtype=np.float64, partial=True, extra_header=[('POLCCONV', 'COSMO')]
        )
        with fits.open(fname, mode='update') as hdul:
            colnames = ['flT', 'flE', 'flB', 'flTE']
            hdul.append(
                fits.BinTableHDU.from_columns(
                    [fits.Column(name=colnames[i], array=_fl, format='D') for i, _fl in enumerate(fl)]
                )
            )
            if eps is not None:
                hdul.append(fits.BinTableHDU.from_columns([fits.Column(name="eps", array=eps, format='D')]))
            hdul.flush()


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


def load_module(module_name, file_path):
    """Load a module from a given file path.

    Parameters
    ----------
    module_name: str
        A designated name for the module (used only internally for namespace consistency). If the name starts
        with "healqest.", then the logging level is set properly as the rest of the healqest modules.
    file_path: str
        Path to the Python file containing the module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dec2tf2d(lx, dec1, dec2):
    """Compute the tf2d (alm) boundary parameters given the SPT field.

    For a given lx cut in a field between dec1 and dec2, the accessible alm space
    is defined by a trapezoid with following exclusions: l < lx; m < m1, and m > k*l.
    This function computes m1 and k (also returns lx) for the trapezoid.

    Parameters
    ----------
    lx: int
        The cut-off multipole of the time-domian filter
    dec1: float
        The "bottom" declination range (higher absolute value)
    dec2: float
        The "top" declination range (higher absolute value)

    Returns
    -------
    lx: int
        l below lx should be 0
    m1: int
        m below m should be 0
    k: float
        m above k* l should be 0. This value only depend on `dec2` and is useful in computing the effective
        transfer function given some m-cut: tf1d = np.sqrt(hp.alm2cl(tf2d)/k).
    """
    # assert 0>dec2>dec1, "dec1 and dec2 should be negative for SPT fields!"
    dec_min = min(abs(dec1), abs(dec2))
    dec_max = min(abs(dec1), abs(dec2))
    if np.sign(dec1) != np.sign(dec2):
        dec_min = 0
    m1 = int(np.floor(lx * np.sin(np.pi / 2 - np.deg2rad(dec_max))))
    k = np.sin(np.pi / 2 - np.deg2rad(dec_min))
    return lx, m1, k


def write_cl(fname, cl, header=None):
    """Write cl into text file."""
    cl = np.atleast_2d(cl)
    lmax = cl.shape[1] - 1
    out = [np.arange(lmax + 1)]
    fmt = ['%6d']
    for i in range(cl.shape[0]):
        fmt.append('%12.5E')
        out.append(cl[i])

    if header is None and cl.shape[0] == 6:
        header = ' ell'
        for _ in ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']:
            header += ' ' * 11 + _
    np.savetxt(fname, np.array(out).T, fmt=fmt, header=header)


def verify_fits(fname, nhdu=1):
    for i in range(nhdu):
        try:
            fits.open(fname, checksum=True, verify='exception')[i + 1].data
        except Exception as e:
            raise e from FileExistsError(f"{fname} is not a valid FITS file or has a corrupted header/data.")
    return True
