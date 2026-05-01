import os
from astropy.io import fits
import numpy as np
import healpy as hp
import importlib
import hashlib
import string
from . import log

logger = log.get_logger(__name__)

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


def map_or_alm(m):
    """Check if object is a map (True) or an alm object."""
    try:
        nside = hp.get_nside(m)
        return True
    except TypeError:
        return False


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


def lowpass_tf(lmax, power=6, lc=6144):
    """
    Compute the low-pass transfer function, in field units, as a function of l.

    Parameters
    ----------
    lmax: int
        The maximum multipole to compute the transfer function for.
    power:
        The power of the low-pass filter. Higher power means a sharper cut-off.
    lc:
        The characteristic multipole of the low-pass filter.

    Returns
    -------
    tf: array
        The low-pass transfer function evaluated at multipoles from 0 to lmax.
    """
    from numpy.polynomial.legendre import leggauss

    N = 30
    t_nodes, weights = leggauss(N)
    t = np.pi * (t_nodes + 1)
    w = np.pi * weights
    x = np.arange(lmax + 1)
    integrand = np.exp(-((x[:, None] * np.sin(t)[None, :] / lc) ** power))
    return integrand @ w / (2 * np.pi)
