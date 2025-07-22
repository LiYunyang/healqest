"""This module provides wrapper of the ducc.sht."""
from functools import cached_property
import os
import warnings
import numpy as np
import healpy as hp
import ducc0
from packaging import version
if version.parse(ducc0.__version__)<version.parse('0.36.0'):
    warnings.warn(
        f"ducc0 version {ducc0.__version__} is lower than the required version 0.36.0. "
        f"The promised performance gain is not guaranteed.", UserWarning,
    )

import logging
logger = logging.getLogger(__name__)

# FIXME: ideally this should not depend on yasolt.
from ducc0.sht import map2leg, leg2map, synthesis_general, adjoint_synthesis_general
# from yasolt.pixelization import Pixelization

rtype = {np.dtype(np.complex128):np.dtype(np.float64), np.dtype(np.complex64):np.dtype(np.float32)}
ctype = {rtype[ctyp]:ctyp for ctyp in rtype}


def get_nthreads(nthreads=None):
    if nthreads is not None and nthreads>0:
        return nthreads
    else:
        return int(os.getenv("OMP_NUM_THREADS", os.cpu_count()))


def _load_pixel_weights(nside):
    from astropy.utils import data
    DATAURL = "https://healpy.github.io/healpy-data/"
    DATAURL_MIRROR = "https://github.com/healpy/healpy-data/releases/download/"
    filename = f"full_weights/healpix_full_weights_nside_{nside:04d}.fits"
    with (data.conf.set_temp("dataurl", DATAURL),
          data.conf.set_temp("dataurl_mirror", DATAURL_MIRROR),
          data.conf.set_temp("remote_timeout", 30)):
        pixel_weights_filename = data.get_pkg_data_filename(filename, package="healpy")
    return pixel_weights_filename


def _load_ring_weights(nside):
    fname = os.path.join(hp.sphtfunc._sphtools.get_datapath(), f"weight_ring_n{nside:05d}.fits")
    return str(fname)


def unfold_weights(nside, ring=True):
    """
    Translated from Healpix/idl/unfold_weights.pro
    """
    from astropy.io import fits
    if ring:
        weights_file = _load_ring_weights(nside)
        w = np.array(fits.open(weights_file)[1].data['TEMPERATURE WEIGHTS']).ravel()
    else:
        weights_file = _load_pixel_weights(nside)
        w = np.array(fits.open(weights_file)[1].data['COMPRESSED PIXEL WEIGHTS'])

    out = np.zeros(hp.nside2npix(nside), dtype=float)

    if ring:
        n4 = 4 * nside
        j = 0
        for i in range(1, n4):
            it = i if i < (n4 - i) else n4 - i
            npix = 4 * (it if it < nside else nside)
            out[j:j + npix] = w[it - 1]
            j += npix
    else:
        npix = hp.nside2npix(nside)
        pix = 0
        vpix = 0
        for i in range(2 * nside):
            shifted = (i < nside - 1) or ((i + nside) % 2 != 0)
            qpix = min(nside, i + 1)
            odd = qpix % 2 != 0
            wpix = ((qpix + 1) // 2) + (0 if (odd or shifted) else 1)
            psouth = npix - pix - (qpix * 4)  # Position in the Southern hemisphere

            for j in range(qpix * 4):
                j4 = j % qpix
                rpix = min(j4, qpix - (1 if shifted else 0) - j4)
                out[pix + j] = w[vpix + rpix]
                if i != 2 * nside - 1:  # Everywhere except the equator
                    out[psouth + j] = w[vpix + rpix]
            pix += qpix * 4
            vpix += wpix
    return out + 1


class Geometry:
    def __init__(self, nside, dec_range=None):
        """
        Setup the geometry for ducc-style SHT.

        Parameters
        ----------
        nside: int
            The nside of the map.
        dec_range: tuple=None
            The range of declination in degrees. If None, use the whole sky.
        """

        n1, n2 = 0, hp.nside2npix(nside)
        if dec_range is not None:
            self.dec_range = dec_range
            t1 = np.pi / 2 - np.deg2rad(max(dec_range))
            t2 = np.pi / 2 - np.deg2rad(min(dec_range))
            n1 = max(n1, hp.ang2pix(nside, t1, 0) - 1)
            n2 = min(n2, hp.ang2pix(nside, t2, 2 * np.pi) + 4*nside+1)
        ipix = np.arange(n1, n2)
        theta, phi = hp.pix2ang(nside, ipix)
        if dec_range is not None:
            sel = np.logical_and(theta > t1, theta < t2)
            theta = theta[sel]
            phi = phi[sel]
            ipix = ipix[sel]

        # this combo is much faster than np.unique, but assumes "theta" is sorted
        assert np.all(np.diff(theta) >= 0)
        _theta = theta[np.insert(np.diff(theta) != 0, 0, True)]
        theta_idx = np.searchsorted(_theta, theta)
        self.nphi = np.bincount(theta_idx)

        _theta = np.asarray(_theta, np.float64)
        self.nphi = np.asarray(self.nphi, np.uint64)

        self.phi0 = np.full(_theta.shape, np.inf, dtype=np.float64)
        np.minimum.at(self.phi0, theta_idx, phi)

        # hardware acceleration for int type but not unsigned int; convert to uint64 later
        self.ofs = np.full(_theta.shape, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(self.ofs, theta_idx, ipix)
        self.ofs = np.asarray(self.ofs, np.uint64)
        # self.weight = np.full(len(_theta), 4 * np.pi / hp.nside2npix(nside), dtype=np.float64)
        self.pixelarea = hp.nside2pixarea(nside)

        self.theta = _theta
        self.nside = nside
        self._lmax = 3 * nside - 1

        self._ring_weights = None
        self._pixel_weights = None

    def restrict(self):
        """
        Create a Geometry instance from nside and optional declination range.
        """
        obj = self.__class__(self.nside, self.dec_range)
        obj.ofs -= self.ofs[0]
        return obj

    @property
    def nph(self):
        return self.nphi

    @property
    def ring_weights(self):
        if self._ring_weights is None:
            self._ring_weights = unfold_weights(self.nside, ring=True)
        return self._ring_weights

    @property
    def pixel_weights(self):
        if self._pixel_weights is None:
            self._pixel_weights = unfold_weights(self.nside, ring=False)
        return self._pixel_weights

    def npix(self):
        """Number of pixels"""
        return int(np.sum(self.nphi))

    def get_kwargs(self, lmax, mmax, nthreads, **kwargs):
        nthreads = get_nthreads(nthreads)
        lmax = self._lmax if lmax is None else lmax
        mmax = lmax if mmax is None else mmax
        out = dict(lmax=lmax, mmax=mmax, nphi=self.nphi, phi0=self.phi0, nthreads=nthreads, ringstart=self.ofs,
                   theta=self.theta)

        if kwargs.get('iter', 0)>0:
            out['maxiter'] = kwargs['iter']
            out['epsilon'] = kwargs['rtol']
        return out

    def format_maps(self, maps, check=True, use_weights=False, use_pixel_weights=False):
        """Formating maps to be compatible as input to `map2alm`.

        The new maps will be reshaped to (nmaps, npix), and masked if there are bad pixels, and applied with
        weights. If no weights is required and the map has no bad pixels, an view of the original map will be
        returned, otherwise a copy will be returned.
        """
        maps = np.atleast_2d(maps)
        mask = None
        masked = None
        if check:
            logger.warning("checking bad pixel for map2alm. this is slow, so set check=False if possible")
            mask = hp.mask_bad(maps)
            masked = np.any(mask)
            if masked:
                maps = maps.copy()
        if use_weights or use_pixel_weights:
            w = 1
            if use_weights:
                w = self.ring_weights
            if use_pixel_weights:
                assert not use_weights
                w = self.pixel_weights
            maps = maps * w.astype(maps.dtype)
        if check:
            if masked:
                maps[mask] = 0
        return maps

    def map2alm(self, maps, lmax=None, mmax=None, iter=0, pol=True, use_weights=False, use_pixel_weights=False,
                nthreads=None,  rtol=1e-5, check=True, alms=None, margin=0, **kwargs):
        """
        Computes the alm of a Healpix map. The input maps must all be in ring ordering.

        Parameters
        maps : array-like, shape (Npix,) or (n, Npix)
            The input map or a list of n input maps. Must be in ring ordering.
        lmax : int, scalar, optional
            Maximum l of the power spectrum. Default: 3*nside-1
        mmax : int, scalar, optional
            Maximum m of the alm. Default: lmax
        iter : int, scalar, optional
            Number of iteration (default: 0). If set to non-zero, this will call pseudo_analysis to solve for the
            alm iteratively.
        pol : bool, optional
            If True, assumes input maps are TQU. Output will be TEB alm's. (input must be 1, 2 or 3 maps)
            If False, apply spin 0 harmonic transform to each map. (input can be any number of maps)
            If there is only one input map, it has no effect. Default: True.
        use_weights: bool, scalar, optional
            If True, use the ring weighting. Default: False.
        use_pixel_weights: bool, optional
            If True, use pixel by pixel weighting, healpy will automatically download the weights, if needed
        nthreads: int=None
            Controls the number of threads used in the computation. If None, it will use the value of
            `OMP_NUM_THREADS`.
        rtol: float, scalar, optional
            Relative tolerance for iterative solution.
        check: bool=True
            Check if there are bad pixels in the input maps and set them to zero for computation (a copy is made
            and the input array is not mutted). If you are certain that there are no bad pixels, you can set this
            to False to save overhead. Default: True.
        alms: array-like
            The output alms buffer.
        """
        assert hp.get_nside(maps) == self.nside
        if margin:
            lmax0 = lmax
            lmax = lmax0 + margin
        else:
            lmax0 = None
        maps = self.format_maps(maps, use_weights=use_weights, use_pixel_weights=use_pixel_weights, check=check)
        nmaps = maps.shape[0]
        # dtype = maps.dtype
        # assert dtype in [np.float32, np.float64]
        # ctype = np.complex64 if dtype == np.float32 else np.complex128
        kw = self.get_kwargs(lmax=lmax, mmax=mmax, nthreads=nthreads, iter=iter, rtol=rtol)
        if alms is None:
            alms = np.zeros((nmaps, hp.Alm.getsize(lmax=kw["lmax"], mmax=kw["mmax"])), dtype=ctype[maps.dtype])
        else:
            if alms.ndim == 1:
                alms = alms[np.newaxis, :]

        func = ducc0.sht.pseudo_analysis if iter else ducc0.sht.adjoint_synthesis
        if pol is False:
            func(map=maps[:, np.newaxis, :], spin=0, alm=alms[:, np.newaxis, :], **kw, **kwargs)
        else:
            assert nmaps in [1, 2, 3], nmaps
            if nmaps in [2, 3]:
                func(map=maps[-2:], spin=2, alm=alms[-2:], **kw, **kwargs)
            if nmaps in [1, 3]:
                # A single map is treated as T-map, since pol=True is the default
                func(map=maps[:1], spin=0, alm=alms[:1], **kw, **kwargs)
        if not iter:
            alms *= self.pixelarea

        out = np.squeeze(alms)
        if lmax0 is not None:
            out = reduce_lmax(out, lmax=lmax0)
        return out

    def alm2map(self, alms, lmax=None, mmax=None, pol=True, nthreads=None, maps=None, **kwargs):
        """
        Computes a Healpix map given the alm.

        The alm are given as a complex array. You can specify lmax and mmax, or they will be computed from array
        size (assuming lmax==mmax).

        Parameters
        ----------
        alms: complex, array or sequence of arrays
            A complex array or a sequence of complex arrays.
            Each array must have a size of the form: mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1
        lmax: None or int, scalar, optional
            Explicitly define lmax (needed if mmax!=lmax)
        mmax: None or int, scalar, optional
            Explicitly define mmax (needed if mmax!=lmax)
        pol: bool, optional
            If True, assumes input alms are TEB. Output will be TQU maps. (input must be 1/2 or 3 alms)
            If False, apply spin 0 harmonic transform to each alm. (input can be any number of alms)
            If there is only one input alm, it has no effect. Default: True.
        nthreads: int=None
            Controls the number of threads used in the computation. If None, it will use the value of
            `OMP_NUM_THREADS`.
        maps: array-like, shape (Npix,) or (nmaps, Npix)
            The output maps buffer. If None, a new array will be created.

        """
        alms = np.atleast_2d(alms)
        nmaps = alms.shape[0]

        # ctype = alms.dtype
        # assert ctype in [np.complex64, np.complex128]
        # dtype = np.float64 if ctype == np.complex128 else np.float32
        if maps is None:
            maps = np.zeros((nmaps, hp.nside2npix(self.nside)), dtype=rtype[alms.dtype])
        else:
            if maps.ndim == 1:
                maps = maps[np.newaxis, :]
        func = ducc0.sht.synthesis
        if lmax is None:
            lmax = hp.Alm.getlmax(alms.shape[-1])

        kw = self.get_kwargs(lmax=lmax, mmax=mmax, nthreads=nthreads)
        if pol is False:
            func(map=maps[:, np.newaxis, :], spin=0, alm=alms[:, np.newaxis, :], **kw, **kwargs)
        else:
            assert nmaps in [1, 2, 3], nmaps
            if nmaps in [2, 3]:
                func(map=maps[-2:], spin=2, alm=alms[-2:], **kw, **kwargs)
            if nmaps in [1, 3]:
                func(map=maps[:1], spin=0, alm=alms[:1], **kw, **kwargs)
        return np.squeeze(maps)

    def map2alm_spin(self, maps, spin, lmax=None, mmax=None, *, iter=0, use_weights=False, use_pixel_weights=False,
                     nthreads=None,  rtol=1e-5, check=True, **kwargs):
        if spin == 0:
            return list(-self.map2alm(maps, lmax=lmax, mmax=mmax, pol=False, iter=iter, use_weights=use_weights,
                                      use_pixel_weights=use_pixel_weights, nthreads=nthreads,  rtol=rtol,
                                      check=check, **kwargs))
        else:
            assert hp.get_nside(maps) == self.nside
            maps = self.format_maps(maps, use_weights=use_weights, use_pixel_weights=use_pixel_weights, check=check)
            nmaps = maps.shape[0]
            assert nmaps == 2, "spin function only accepts 2 maps"
            # dtype = maps.dtype
            # assert dtype in [np.float32, np.float64]
            # ctype = np.complex64 if dtype == np.float32 else np.complex128
            kw = self.get_kwargs(lmax=lmax, mmax=mmax, nthreads=nthreads, iter=iter, rtol=rtol)
            func = ducc0.sht.pseudo_analysis if iter else ducc0.sht.adjoint_synthesis
            alms = np.zeros((nmaps, hp.Alm.getsize(lmax=kw["lmax"], mmax=kw["mmax"])), dtype=ctype[maps.dtype])
            func(map=maps, spin=np.abs(spin), alm=alms, **kw, **kwargs)
            if not iter:
                alms *= self.pixelarea
            return list(alms)

    def alm2map_spin(self, alms, spin, lmax=None, mmax=None, *, nthreads=None, maps=None, **kwargs):
        if spin ==0:
            out = list(-self.alm2map(alms, lmax=lmax, mmax=mmax, pol=False, nthreads=nthreads, maps=maps, **kwargs))
            if maps is not None:
                # inplace mutation if maps is sent in as a buffer.
                maps *= -1
            return out
        else:
            alms = np.atleast_2d(alms)
            nmaps = alms.shape[0]
            assert nmaps == 2, "spin function only accepts 2 maps"
            # ctype = alms.dtype
            # assert ctype in [np.complex64, np.complex128]
            # dtype = np.float64 if ctype == np.complex128 else np.float32
            if maps is None:
                maps = np.zeros((nmaps, hp.nside2npix(self.nside)), dtype=rtype[alms.dtype])
            else:
                assert maps.shape[0] == 2
            func = ducc0.sht.synthesis
            if lmax is None:
                lmax = hp.Alm.getlmax(alms.shape[-1])
            kw = self.get_kwargs(lmax=lmax, mmax=mmax, nthreads=nthreads)
            func(map=maps, spin=np.abs(spin), alm=alms, **kw, **kwargs)
            return list(maps)

    def smoothing(self, maps_in, fwhm=0.0, sigma=None, beam_window=None, pol=True, iter=0, lmax=None, mmax=None,
                  use_weights=False, use_pixel_weights=False, nthreads=None, check=False):
        if check:
            masks = hp.mask_bad(maps_in)
            maps_in[masks] = 0
        alms = self.map2alm(maps_in, pol=pol, lmax=lmax, mmax=mmax, iter=iter, use_weights=use_weights,
                            use_pixel_weights=use_pixel_weights, nthreads=nthreads, check=False)
        alms = hp.smoothalm(alms, fwhm=fwhm, sigma=sigma, beam_window=beam_window, pol=pol, mmax=mmax, inplace=True)
        maps_out = self.alm2map(alms, pol=pol, nthreads=nthreads)
        if check:
            maps_out[masks] = hp.UNSEEN
        return maps_out


class Pixelization(object):
    def __init__(self, loc: np.ndarray = None, geom=None, epsilon=1e-7):
        """Attempt at a unified interface for pixelizations and filters

        Parameters
        ----------
        loc: array of locations (npix x 2) in (tht=colatitude, phi=longitude) in radians
        geom: isolatitude pixelization Geom instance
        epsilon: (optional, only relevant in the 'loc' case) Accuracy of the *_general* SHTs

        Only one of the two can be set
        """
        assert loc or geom
        assert geom is None or loc is None, "only one of loc or geom can be set"

        if loc is not None:
            thtmin = np.min(loc[:, 0])
            thtmax = np.max(loc[:, 0])
            npix = loc.shape[0]
        else:
            thtmin = np.min(geom.theta)
            thtmax = np.max(geom.theta)
            npix = geom.npix()

        self.loc = loc
        self.geom = geom
        self.thtrange = (thtmin, thtmax)
        self.epsilon = epsilon
        self._npix = npix

    def npix(self):
        return self._npix

    def synthesis(self, alm: np.ndarray, spin: int, lmax: int, mmax: int, nthreads: int, m: np.ndarray = None,
                  **kwargs):
        """Produces a map or a pair of maps from alm array

        Parameters
        ----------
        alm: shape (ncomp, alm_size) where ncomp is 1 or 2 (gradient and curl components)
        spin: int
            spin of the field
        lmax: int
            maximum l of the alm layout
        mmax: maximum m of the alm layout
        nthreads: number of threads to use
        m: map array of shape (1 + (spin > 0), npix) (initialized if not provided)

        Returns
        -------
        m
        """
        if m is None:
            m = np.empty((1 if spin == 0 else 2, self.npix()), dtype=rtype[alm.dtype])
        if self.loc is not None:
            return self._synthesis_loc(alm, spin, lmax, mmax, nthreads, m, **kwargs)
        return self._synthesis_geom(alm, spin, lmax, mmax, nthreads, m, **kwargs)

    def adjoint_synthesis(self, alm: np.ndarray, spin: int, lmax: int, mmax: int, nthreads: int, m: np.ndarray,
                          **kwargs):
        """Adjoint (not inverse) operation of alm to map (synthesis)

        Parameters
        ----------
        alm: shape (ncomp, alm_size) where ncomp is 1 or 2 (gradient and curl components)
        spin: spin of the field
        lmax: maximum l of the alm layout
        mmax: maximum m of the alm layout
        nthreads: number of threads to use
        m: map array of shape (1 + (spin > 0), npix)

        Returns
        -------
        alm
        """
        if self.loc is not None:
            return self._adjoint_synthesis_loc(alm, spin, lmax, mmax, nthreads, m, **kwargs)
        return self._adjoint_synthesis_geom(alm, spin, lmax, mmax, nthreads, m, **kwargs)

    def _synthesis_geom(self, alm: np.ndarray, spin: int, lmax: int, mmax: int, nthreads: int, m: np.ndarray,
                        **kwargs):
        # relevant kwargs here: mode.
        assert self.geom is not None, 'no isolatitude geometry set'
        return self.geom.synthesis(alm, spin, lmax, mmax, nthreads, map=m, **kwargs)

    def _adjoint_synthesis_geom(self, alm: np.ndarray, spin: int, lmax: int, mmax: int, nthreads: int,
                                m: np.ndarray, **kwargs):
        # relevant kwargs here: mode.
        assert self.geom is not None, 'no isolatitude geometry set'
        return self.geom.adjoint_synthesis(np.atleast_2d(m), spin, lmax, mmax, nthreads, alm=np.atleast_2d(alm),
                                           apply_weights=False, **kwargs)

    def _synthesis_loc(self, alm: np.ndarray, spin: int, lmax: int, mmax: int, nthreads: int, m: np.ndarray,
                       **kwargs):
        assert self.loc is not None, 'no locations set'
        return synthesis_general(map=m, lmax=lmax, mmax=mmax, alm=alm, loc=self.loc, spin=spin, nthreads=nthreads,
                                 epsilon=self.epsilon, **kwargs)

    def _adjoint_synthesis_loc(self, alm: np.ndarray, spin: int, lmax: int, mmax: int, nthreads: int, m: np.ndarray,
                               **kwargs):
        assert self.loc is not None, 'no locations set'
        return adjoint_synthesis_general(map=np.atleast_2d(m), lmax=lmax, mmax=mmax, alm=np.atleast_2d(alm),
                                         loc=self.loc, spin=spin, nthreads=nthreads, epsilon=self.epsilon, **kwargs)


def st2mmax(spin, tht, lmax):
    """
    Converts spin, tht and lmax to a maximum effective m.

    According to libsharp paper polar optimization formula Eqs. 7-8.
    For a given mmax, one needs then in principle 2 * mmax + 1 longitude points for exact FFT's
    """

    T = max(0.01 * lmax, 100)
    b = - 2 * spin * np.cos(tht)
    c = -(T + lmax * np.sin(tht)) ** 2 + spin ** 2
    mmax = 0.5 * (- b + np.sqrt(b * b - 4 * c))
    return mmax


def map2lens(maps, plm, g=None, **kwargs):
    """
    Compute the first order lensing distortion of the input maps.

    Parameters
    ----------
    maps : array-like, shape (3, Npix)
        Unlensed CMB maps in TQU ordering.
    plm : array-like
        Alm of the lensing potential phi.
    g: Geometry
    """
    lmax = hp.Alm.getlmax(len(plm))
    nside = hp.get_nside(maps)
    if g is None:
        alms = hp.map2alm(maps, lmax=lmax, iter=0, **kwargs)
    else:
        assert g.nside==nside
        alms = g.map2alm(maps, lmax=lmax, iter=0, **kwargs)

    return alm2lens(alms, plm, nside=nside, g=g, **kwargs)


def alm2lens(alms, plm, nside, g=None, **kwargs):
    """
    Same as map2lens, but takes alm as input.

    Parameters
    ----------
    alms : array-like
        Unlensed CMB map alms in TEB ordering.
    plm : array-like
        Alm of the lensing potential phi.
    nside: int
        Nside of the output maps.
    g: Geometry
    """
    lmax = hp.Alm.getlmax(len(plm))
    ell = np.arange(lmax + 1)
    al0 = -np.sqrt(ell * (ell + 1) / 2)

    if g is None:
        alm2map_spin = hp.alm2map_spin
        kwargs['nside'] = nside
    else:
        assert g.nside==nside
        alm2map_spin = g.alm2map_spin

    zero = np.zeros_like(plm)
    p = alm2map_spin([hp.almxfl(plm, -al0), zero], spin=1, lmax=lmax, **kwargs)

    alms = np.atleast_2d(alms)
    tp = alm2map_spin([hp.almxfl(alms[0], -al0), zero], spin=1, lmax=lmax, **kwargs)
    out = np.zeros((alms.shape[0], hp.nside2npix(nside)), dtype=float)
    out[0] = 2 * (p[0] * tp[0] + p[1] * tp[1])
    del tp

    if alms.shape[0]==3:
        al2p = -np.sqrt((ell - 2) * (ell + 3) / 2)
        Zp = alm2map_spin([hp.almxfl(alms[1], al2p),
                           hp.almxfl(alms[2], al2p)], spin=3, lmax=lmax, **kwargs)

        al2m = -np.sqrt((ell + 2) * (ell - 1) / 2)
        Zm = alm2map_spin([hp.almxfl(alms[1], al2m),
                           hp.almxfl(alms[2], al2m)], spin=1, lmax=lmax, **kwargs)

        out[1] = (p[1] * Zm[1] - p[0] * Zm[0]) + (p[0] * Zp[0] + p[1] * Zp[1])
        out[2] = -(p[0] * Zm[1] + p[1] * Zm[0]) + (p[0] * Zp[1] - p[1] * Zp[0])
    return out


def get_dec_range(mask, dec=None):
    """
    Find the proper decrange for ducc wrapper for a given mask.
    """
    if isinstance(mask, str):
        mask = hp.read_map(mask)
    nside = hp.get_nside(mask)
    if dec is None:
        dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)[1]

    dec1 = int(np.floor(np.min(dec[mask > 0])))
    dec2 = int(np.ceil(np.max(dec[mask > 0])))
    return dec1, dec2


def reduce_lmax(alm, lmax=4000):
    """
    Reduce the lmax of input alm

    Copied from healqest_utils.py to avoid circular dependency within healqest.
    """
    lmaxin = hp.Alm.getlmax(alm.shape[-1])
    logger.debug(f"Reducing lmax: lmax_in={lmaxin} -> lmax_out={lmax}")
    almout = np.zeros((*alm.shape[:-1], hp.Alm.getsize(lmax)), dtype=alm.dtype)
    oldi = 0
    newi = 0
    dl = lmaxin - lmax
    for i in range(0, lmax + 1):
        oldf = oldi + lmaxin + 1 - i
        newf = newi + lmax + 1 - i
        almout[..., newi:newf] = alm[..., oldi:oldf - dl]
        oldi = oldf
        newi = newf
    return almout


class BtNiB_light:
    def __init__(self, pixels, job: str, lmmax: tuple[int, int],
                 nthreads=0, r_dtype=np.float64,
                 tf_params={'tf_geo': None, 'tf_pixs_in': None, 'tf_pixs_out': None, 'locs_in': None, 'lx_cut': 0,
                            'm_cut': 0, 'tf_psi': None},
                 b_mmin=0, loc_c=None):
        """Inverse noise matrix operator, combining beam and inverse noise variance maps for a number of frequency channels

            Args:
                pixels: generalized geometry object (see Pixelization.py) that performs the SHT's and their adjoint
                mode: 'GRAD_ONLY' or 'STANDARD' (ducc SHT modes. Set this to "GRAD_ONLY" if it only takes the gradient mode as inputs)
                lmmax: alm array layout
                nthreads: number of threads for SHTs
                r_dtype: precision of the computation
                b_mmin, loc_c: (optional) The transfer functions rotate back from pole to loc_c and sets to zero m < mmin. This slows things down by a significant amount if it actually must be rotated)

            Note:
                For now, the pixelization is the same across channels, but this could be adapted.

                This performs (a channel, i pixel, lm harmonics)

                    :math:`\sum_{a, i, lm} b_l'Y^{\dagger}_{l'm'}(\hn_i) N^{-1}_i /s^2_a Y_{lm}(\hat n_i) b^a_l a_{lm}`

                    (plus rotation - alm masking in each b_l operation if b_mmin and loc_c are set)

        """
        assert job in ['T', 'EB', 'TEB'], job
        mode = 'STANDARD'

        # SO FAR each channel has the same job (T, EB, TEB)
        # TODO: have a to have a list of spins for each channels, and then just adapt the tf to loop over the
        self.nchannels = 1
        self.nalms = len(job)
        self.nmaps = self.job2spins(job)[1]
        self.job = job
        self.mode = mode

        self.pixels = pixels

        self.nthreads = get_nthreads(nthreads)

        self.lmmax = lmmax

        self.dtype = r_dtype
        self._alms = None
        self._maps = None
        self._tfgeo_maps = None

        # This is for a clean theta-independdent m-cut after rotation (to be dropped)
        self.b_mmin = b_mmin
        self.loc_c = loc_c

        # Transfer function parameters #FIXME: this is a mess now
        self.tf_geo = tf_params.get('tf_geo', None)
        self.tf_pixs_in = tf_params.get('tf_pixs_in', None)
        self.tf_pixs_out = tf_params.get('tf_pixs_out', self.tf_pixs_in)
        self.lx_cut = tf_params.get('lx_cut', 0)  # to use a tht-dependent m-cut
        self.m_cut = tf_params.get('m_cut', 0)  # to apply a fixed-m cut
        self.m_apodeg = tf_params.get('m_apodeg', 0.)
        self.tf_locs_in = tf_params.get('locs_in', None)

        # FIXME: this will only work if  pixs_in are the same than pixs_out
        self.tf_psi = tf_params.get('tf_psi', None)
        self._has_tf_psi = (self.tf_psi is not None and np.any(self.tf_psi))
        self.pixels_tf = Pixelization(loc=self.tf_locs_in,
                                      epsilon=self.pixels.epsilon) if self.tf_locs_in is not None else self.pixels

        if self.tf_geo is not None:  # gets mmin - mmax boundaries
            assert self.tf_pixs_in is not None
            assert tf_params.get('lx_cut', None) is not None
            assert self.tf_pixs_out.size==self.pixels.npix(), (self.tf_pixs_out.size, self.pixels.npix())
            assert self.m_apodeg==0, 'need to reimplement that, it was not working well anyways'
            ms_min = np.int_(self.lx_cut*np.sin(np.minimum(self.tf_geo.theta + self.m_apodeg*np.pi/180., np.pi)))
            ms_max = np.int_(self.lx_cut*np.sin(np.maximum(self.tf_geo.theta - self.m_apodeg*np.pi/180., 0.)))
            mcuts_min = np.maximum(np.minimum(ms_min, ms_max), self.m_cut)
            mcuts_max = np.maximum(np.maximum(ms_min, ms_max), self.m_cut)
            mcuts_min = np.maximum(mcuts_min, 0)
            self.mi = mcuts_min
            self.ma = mcuts_max

        if self.pixels_tf.npix()!=self.pixels.npix():  # Sanity checks
            assert self.pixels.npix()==self.tf_pixs_out.size, (self.pixels_tf.npix(), self.tf_pixs_out.size)
            assert self.pixels_tf.npix()==self.tf_pixs_in.size, (self.pixels_tf.npix(), self.tf_pixs_in.size)
            print('Transfer: ')
            print('  Number of in  pixels: %s'%self.tf_pixs_in.size)
            print('  Number of out pixels: %s'%self.tf_pixs_out.size)

    def _cut_ms(self, legs):
        # TODO:
        ms = np.arange(legs.shape[2], dtype=int)
        for ir, (mc, np_i) in enumerate(zip(self.ma, 1./self.tf_geo.nph)):
            legs[:, ir, :] *= np_i*(ms<mc)

    def _synthesize_channel(self, pixels: Pixelization, channel, alms, maps, subjob=None):
        subjob = subjob or self.job
        assert self.mode=='STANDARD', 'Only STANDARD mode implemented'
        spins, ncomp = self.job2spins(subjob)
        if maps is None:
            nmaps = np.sum([1 + (spin>0) for spin in spins])
            maps = np.empty((nmaps, pixels.npix()), dtype=self.dtype)
        assert alms.shape==(ncomp, hp.Alm.getsize(*self.lmmax)), (alms.shape, ncomp, hp.Alm.getsize(*self.lmmax))
        assert maps.shape==(ncomp, pixels.npix())
        i = 0
        for spin in spins:
            ncomp = 1 + (spin>0)
            pixels.synthesis(alms[i:i + ncomp], spin, self.lmmax[0], self.lmmax[1], self.nthreads,
                             m=maps[i:i + ncomp], mode=self.mode)
            i += ncomp
        return maps

    def _adjoint_synthesize_channel(self, pixels, channel, alms, maps, subjob=None):
        subjob = subjob or self.job
        assert self.mode=='STANDARD', 'Only STANDARD mode implemented'
        spins, ncomp = self.job2spins(subjob)
        assert alms.shape==(ncomp, hp.Alm.getsize(*self.lmmax))
        assert maps.shape==(ncomp, pixels.npix())
        i = 0
        for spin in spins:
            ncomp = 1 + (spin>0)
            pixels.adjoint_synthesis(alms[i:i + ncomp], spin, self.lmmax[0], self.lmmax[1], self.nthreads,
                                     m=maps[i:i + ncomp], mode=self.mode)
            i += ncomp
        return alms

    @staticmethod
    def job2spins(job):
        assert job in ['T', 'EB', 'TEB'], job
        # spins = [0]*('T' in subjob) + [2]*('EB' in subjob)
        # ncomp = np.sum([1 + (spin>0) for spin in spins])
        if job=='T':
            spins, ncomp = [0], 1
        elif job=='EB':
            spins, ncomp = [2], 2
        elif job=='TEB':
            spins, ncomp = [0, 2], 3
        else:
            raise ValueError(f"only supported jobs are 'T', 'EB' and 'TEB', got {job}")
        return spins, ncomp

    def _rotate(self, maps, sgn, subjob=None):
        # TODO: use usual complex view etc ?
        # TODO: better, include psi in loc-based pixelizations object?
        # YL: this function is effectivly unused currently. (tf_psi=0)
        subjob = subjob or self.job
        spins, ncomp = self.job2spins(subjob)
        i = 0
        for spin in spins:
            ncomp = 1 + (spin>0)
            if spin:
                assert self.tf_psi is not None
                if self._has_tf_psi:
                    maps_c = (maps[i] + 1j*maps[i + 1])*np.exp((1j*sgn*spin)*self.tf_psi)
                    maps[i] = maps_c.real
                    maps[i + 1] = maps_c.imag
            i += ncomp

    def _apply_tf_inplace(self, channel, alms, maps, subjob=None):
        assert maps.ndim==2, maps.shape

        if self.lx_cut>0 or self.m_cut>0:
            geo = self.tf_geo
            if not (self.tf_pixs_in is self.tf_pixs_out):
                self._tfgeo_maps[:maps.shape[0], :] = 0.
            tmps = self._synthesize_channel(self.pixels_tf, channel, alms, None, subjob=subjob)
            self._rotate(tmps, +1, subjob=subjob)
            self._tfgeo_maps[:maps.shape[0], self.tf_pixs_in] = tmps
            legs = map2leg(map=self._tfgeo_maps[:maps.shape[0]], nphi=geo.nph, ringstart=geo.ofs,
                           mmax=np.max(self.ma), phi0=geo.phi0, nthreads=self.nthreads)
            self._cut_ms(legs)

            # removes the low-m part
            maps[:] = self._tfgeo_maps[:maps.shape[0], self.tf_pixs_out] - leg2map(leg=legs, nphi=geo.nph,
                                                                                   ringstart=geo.ofs,
                                                                                   phi0=geo.phi0,
                                                                                   nthreads=self.nthreads)[:,self.tf_pixs_out]
            self._rotate(maps, -1, subjob=subjob)
        else:
            self._synthesize_channel(self.pixels, channel, alms, maps, subjob=subjob)

    def _apply_tf_adjoint_inplace(self, channel, alms, maps, subjob=None):
        assert maps.ndim==2, maps.shape
        if self.lx_cut>0 or self.m_cut>0:
            self.allocate_maps()
            geo = self.tf_geo
            if not (self.tf_pixs_in is self.tf_pixs_out):
                self._tfgeo_maps[:maps.shape[0], :] = 0.
            self._rotate(maps, +1, subjob=subjob)
            self._tfgeo_maps[:maps.shape[0], self.tf_pixs_out] = maps
            legs = map2leg(map=self._tfgeo_maps[:maps.shape[0]], nphi=geo.nph, ringstart=geo.ofs,
                           mmax=np.max(self.ma), phi0=geo.phi0, nthreads=self.nthreads)
            self._cut_ms(legs)
            m = self._tfgeo_maps[:maps.shape[0], self.tf_pixs_in] - leg2map(leg=legs, nphi=geo.nph,
                                                                            ringstart=geo.ofs, phi0=geo.phi0,
                                                                            nthreads=self.nthreads)[:,
                                                                    self.tf_pixs_in]
            self._rotate(m, -1, subjob=subjob)
            self._adjoint_synthesize_channel(self.pixels_tf, channel, alms, m, subjob=subjob)
        else:
            self._adjoint_synthesize_channel(self.pixels, channel, alms, maps, subjob=subjob)

    def allocate_maps(self):
        # NB: we can use the same maps for each channel
        if self._maps is None:
            self._maps = np.empty((np.max(self.nmaps), self.pixels.npix()), dtype=self.dtype)
        if self._tfgeo_maps is None and max(self.lx_cut, self.m_cut) > 0:
            self._tfgeo_maps = np.zeros((np.max(self.nmaps), self.tf_geo.npix()), dtype=self.dtype)

    def deallocate(self):
        self._alms = None
        self._maps = None
        self._tfgeo_maps = None

    def apply(self, alms):
        subjob = self.job
        nalms = len(subjob)
        assert alms.shape==(nalms, hp.Alm.getsize(*self.lmmax)), (alms.shape, nalms, hp.Alm.getsize(*self.lmmax))
        self.allocate_maps()
        _maps = self._maps[:nalms]
        self._apply_tf_inplace(0, alms, _maps, subjob=subjob)
        self._apply_tf_adjoint_inplace(0, alms, _maps, subjob=subjob)


class GeometryTF:
    def __init__(self, geom, ipix, lx_cut=0, m_cut=0, m_apodeg=0):
        assert geom.ofs[0] == np.min(geom.ofs)
        self.g = geom
        self.lx_cut = lx_cut
        self.m_cut = m_cut
        self.m_apodeg = m_apodeg
        self.ipix = ipix
        self.tf_pix = ipix-self.g.ofs[0]  # internal index to map from reduced pix to full pixels.

        assert self.m_apodeg==0, 'need to reimplement that, it was not working well anyways'
        ms_min = np.int_(self.lx_cut*np.sin(np.minimum(self.g.theta + self.m_apodeg*np.pi/180., np.pi)))
        ms_max = np.int_(self.lx_cut*np.sin(np.maximum(self.g.theta - self.m_apodeg*np.pi/180., 0.)))
        mcuts_min = np.maximum(np.minimum(ms_min, ms_max), self.m_cut)
        mcuts_max = np.maximum(np.maximum(ms_min, ms_max), self.m_cut)
        mcuts_min = np.maximum(mcuts_min, 0)
        self.mi = mcuts_min
        self.ma = mcuts_max

    @cached_property
    def ofs(self):
        return self.g.ofs-self.g.ofs[0]

    @cached_property
    def npix(self):
        return self.g.npix()

    def _cut_ms(self, legs):
        ms = np.arange(legs.shape[2], dtype=int)
        for ir, (mc, np_i) in enumerate(zip(self.ma, 1./self.g.nphi)):
            legs[:, ir, :] *= np_i*(ms<mc)


    def _apply_tf_inplace(self, alms, maps, nthreads):
        if maps is not None:
            assert maps.ndim==2, maps.shape
        assert alms.ndim==2, alms.shape
        self.g.alm2map(alms, maps=maps, nthreads=nthreads, )

        # if self.lx_cut>0 or self.m_cut>0:
        #     _maps = np.zeros((maps.shape[0], self.npix), dtype=maps.dtype)
        #     _maps[:, self.tf_pix] = maps[:, self.ipix]
        #     legs = map2leg(map=_maps, nphi=self.g.nphi, ringstart=self.ofs, mmax=np.max(self.ma), phi0=self.g.phi0,
        #                    nthreads=nthreads)
        #     self._cut_ms(legs)
            # removes the low-m part
            # cut_map = leg2map(leg=legs, nphi=self.g.nph, ringstart=self.ofs, phi0=self.g.phi0, nthreads=nthreads)
            # maps[:, self.ipix] -= cut_map[:, self.tf_pix]
        self._apply_map(maps, nthreads=nthreads)
        # return maps

    def _apply_tf_adjoint_inplace(self, alms, maps, lmax, nthreads):
        assert maps.ndim==2, maps.shape
        if alms is not None:
            assert alms.ndim==2, alms.shape
        # if self.lx_cut>0 or self.m_cut>0:
        #     _maps = np.zeros((maps.shape[0], self.npix), dtype=maps.dtype)
        #     _maps[:, self.tf_pix] = maps[:, self.ipix]
        #     legs = map2leg(map=_maps, nphi=self.g.nph, ringstart=self.ofs, mmax=np.max(self.ma), phi0=self.g.phi0,
        #                    nthreads=nthreads)
        #     self._cut_ms(legs)
        #     cut_map = leg2map(leg=legs, nphi=self.g.nphi, ringstart=self.ofs, phi0=self.g.phi0, nthreads=nthreads)
        #     maps[:, self.ipix] -= cut_map[:, self.tf_pix]
        maps = self._apply_map(maps, nthreads=nthreads)
        alms = self.g.map2alm(maps, alms=alms, lmax=lmax, nthreads=nthreads, check=False)
        return alms

    def _apply_map(self, maps, nthreads=None):
        """
        Parameters
        ----------
        maps: array-like, shape (ncomp, 12*nside**2)
        """
        nthreads = get_nthreads(nthreads)
        if self.lx_cut>0 or self.m_cut>0:
            _maps = np.zeros((maps.shape[0], self.npix), dtype=maps.dtype)
            _maps[:, self.tf_pix] = maps[:, self.ipix]
            legs = map2leg(map=_maps, nphi=self.g.nphi, ringstart=self.ofs, mmax=np.max(self.ma), phi0=self.g.phi0,
                           nthreads=nthreads)
            self._cut_ms(legs)
            cut_map = leg2map(leg=legs, nphi=self.g.nphi, ringstart=self.ofs, phi0=self.g.phi0, nthreads=nthreads)
            print(cut_map.shape)
            maps[:, self.ipix] -= cut_map[:, self.tf_pix]
        return maps

    def apply(self, alms, nthreads=None):
        alms = np.atleast_2d(alms)
        lmax = hp.Alm.getlmax(alms.shape[-1])
        nmaps = alms.shape[0]
        _maps = np.zeros((nmaps, hp.nside2npix(self.g.nside)), dtype=rtype[alms.dtype])
        nthreads = get_nthreads(nthreads)
        _maps = self._apply_tf_inplace(alms, _maps, nthreads=nthreads)
        alms = self._apply_tf_adjoint_inplace(alms, _maps, lmax=lmax, nthreads=nthreads)
        return alms

