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
        """alias for consistency with lenspyx-based calls"""
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


class GeometryTF:
    """Extends the Geometry class to apply a (theta-dependent) filter."""
    def __init__(self, geom, ipix=None, lx_cut=0, m_cut=0, m_apodeg=0):
        assert geom.ofs[0] == np.min(geom.ofs)
        self.g = geom
        self.lx_cut = lx_cut
        if m_cut is None:
            m_cut = 0
        if m_apodeg is None:
            m_apodeg = 0
        self.m_cut = m_cut
        self.m_apodeg = m_apodeg
        self.ipix = None
        self.tf_pix = None
        if ipix is not None:
            self.set_ipix(ipix)

        assert self.m_apodeg==0, 'need to reimplement that, it was not working well anyways'
        ms_min = np.int_(self.lx_cut*np.sin(np.minimum(self.g.theta + self.m_apodeg*np.pi/180., np.pi)))
        ms_max = np.int_(self.lx_cut*np.sin(np.maximum(self.g.theta - self.m_apodeg*np.pi/180., 0.)))
        mcuts_min = np.maximum(np.minimum(ms_min, ms_max), self.m_cut)
        mcuts_max = np.maximum(np.maximum(ms_min, ms_max), self.m_cut)
        mcuts_min = np.maximum(mcuts_min, 0)
        self.mi = mcuts_min
        self.ma = mcuts_max

    def set_ipix(self, ipix):
        """delayed setting of the pixels"""
        self.ipix = ipix
        self.tf_pix = ipix - self.g.ofs[0]  # internal index to map from reduced pix to full pixels.
        assert all(self.tf_pix >= 0)

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
        maps = self.apply_map(maps, nthreads=nthreads)
        return maps

    def _apply_tf_adjoint_inplace(self, alms, maps, lmax, nthreads):
        assert maps.ndim==2, maps.shape
        if alms is not None:
            assert alms.ndim==2, alms.shape
        maps = self.apply_map(maps, nthreads=nthreads)
        alms = self.g.map2alm(maps, alms=alms, lmax=lmax, nthreads=nthreads, check=False)
        return alms

    def apply_map(self, maps, nthreads=None):
        """
        Parameters
        ----------
        maps: array-like, shape (ncomp, 12*nside**2)
        nthreads: int=None
        """
        assert maps.ndim==2, maps.shape
        nmaps, npix = maps.shape
        nthreads = get_nthreads(nthreads)
        kw = dict(nphi=self.g.nphi, ringstart=self.ofs, phi0=self.g.phi0, nthreads=nthreads)
        if self.lx_cut>0 or self.m_cut>0:
            # if nmaps <= 3 and maps.dtype == np.float64:
            #     _maps = self.buffer3d[:nmaps]
            # else:
            _maps = np.zeros((nmaps, self.npix), dtype=maps.dtype)
            _maps[:, self.tf_pix] = maps[:, self.ipix]
            legs = ducc0.sht.map2leg(map=_maps, mmax=np.max(self.ma), **kw)
            self._cut_ms(legs)
            cut_map = ducc0.sht.leg2map(leg=legs, **kw)
            maps[:, self.ipix] -= cut_map[:, self.tf_pix]
        return maps

    def apply(self, alms, nthreads=None):
        """Example of how to filter the maps starting from alm and back to alm."""
        alms = np.atleast_2d(alms)
        lmax = hp.Alm.getlmax(alms.shape[-1])
        nmaps = alms.shape[0]
        _maps = np.zeros((nmaps, hp.nside2npix(self.g.nside)), dtype=rtype[alms.dtype])
        nthreads = get_nthreads(nthreads)
        _maps = self._apply_tf_inplace(alms, _maps, nthreads=nthreads)
        alms = self._apply_tf_adjoint_inplace(alms, _maps, lmax=lmax, nthreads=nthreads)
        return alms

    def filter_alms(self, alms, nthreads=None):
        """Apply lx cut to alms of shape (1,2,3) for T/EB/TEB. [inplace]"""
        nthreads = get_nthreads(nthreads)
        lmax = hp.Alm.getlmax(alms.shape[-1])
        maps = np.atleast_2d(self.g.alm2map(alms, nthreads=nthreads))
        self.apply_map(maps, nthreads=nthreads)
        return self.g.map2alm(maps, nthreads=nthreads, check=False, lmax=lmax)

    def filter_maps(self, maps, nthreads=None):
        """Apply lx cut to maps of shape (1,2,3) for T/QU/TQU. [inplace]"""
        return self.apply_map(np.atleast_2d(maps), nthreads=nthreads)

    def filter_maps_partial(self, maps, nthreads=None):
        """"fast filtering of a 1d partial maps (only pixels defined by self.ipix)"""
        nthreads = get_nthreads(nthreads)
        assert maps.ndim == 1
        kw = dict(nphi=self.g.nphi, ringstart=self.ofs, phi0=self.g.phi0, nthreads=nthreads)
        if self.lx_cut>0 or self.m_cut>0:
            _maps = self.buffer
            _maps[0, self.tf_pix] = maps
            legs = ducc0.sht.map2leg(map=_maps, mmax=np.max(self.ma), **kw)
            self._cut_ms(legs)
            cut_map = ducc0.sht.leg2map(leg=legs, **kw).squeeze(axis=0)
            maps -= cut_map[self.tf_pix]
        return maps

    @cached_property
    def buffer(self):
        return np.zeros((1, self.npix), dtype=np.float64)

    @cached_property
    def buffer3d(self):
        return np.zeros((3, self.npix), dtype=np.float64)


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
