"""This module provides wrapper of the ducc.sht."""
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

        # hardware acceleration for int type but not unsigned int convert to uint64 later
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

    def get_kwargs(self, lmax, mmax, nthreads, **kwargs):
        if nthreads is None:
            nthreads = int(os.getenv("OMP_NUM_THREADS", os.cpu_count()))
        lmax = self._lmax if lmax is None else lmax
        mmax = lmax if mmax is None else mmax
        out = dict(lmax=lmax, mmax=mmax, nphi=self.nphi, phi0=self.phi0, nthreads=nthreads, ringstart=self.ofs,
                   theta=self.theta)

        if kwargs.get('iter', 0)>0:
            out['maxiter'] = kwargs['iter']
            out['epsilon'] = kwargs['rtol']
        return out

    def map2alm(self, maps, lmax=None, mmax=None, iter=0, pol=True, use_weights=False, use_pixel_weights=False,
                nthreads=None,  rtol=1e-5, **kwargs):
        assert hp.get_nside(maps) == self.nside

        maps = np.atleast_2d(maps)
        nmaps = maps.shape[0]
        dtype = maps.dtype
        assert dtype in [np.float32, np.float64]

        w = None
        if use_weights:
            w = self.ring_weights
        if use_pixel_weights:
            assert not use_weights
            w = self.pixel_weights
        if w is not None:
            maps *= w.astype(dtype)

        ctype = np.complex64 if dtype == np.float32 else np.complex128
        kw = self.get_kwargs(lmax=lmax, mmax=mmax, nthreads=nthreads, iter=iter, rtol=rtol)
        alms = np.zeros((nmaps, hp.Alm.getsize(lmax=kw["lmax"], mmax=kw["mmax"])), dtype=ctype)

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
        return np.squeeze(alms)

    def alm2map(self, alms, lmax=None, mmax=None, pol=True, nthreads=None, maps=None, **kwargs):
        alms = np.atleast_2d(alms)
        nmaps = alms.shape[0]

        ctype = alms.dtype
        assert ctype in [np.complex64, np.complex128]
        dtype = np.float64 if ctype == np.complex128 else np.float32
        if maps is None:
            maps = np.zeros((nmaps, hp.nside2npix(self.nside)), dtype=dtype)
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
                     nthreads=None,  rtol=1e-5, **kwargs):
        if spin == 0:
            return list(-self.map2alm(maps, lmax=lmax, mmax=mmax, pol=False, iter=iter, use_weights=use_weights,
                                      use_pixel_weights=use_pixel_weights, nthreads=nthreads,  rtol=rtol, **kwargs))
        else:
            assert hp.get_nside(maps) == self.nside

            maps = np.atleast_2d(maps)
            nmaps = maps.shape[0]
            assert nmaps == 2, "spin function only accepts 2 maps"
            dtype = maps.dtype
            assert dtype in [np.float32, np.float64]

            w = None
            if use_weights:
                w = self.ring_weights
            if use_pixel_weights:
                assert not use_weights
                w = self.pixel_weights
            if w is not None:
                maps *= w.astype(dtype)

            ctype = np.complex64 if dtype == np.float32 else np.complex128
            kw = self.get_kwargs(lmax=lmax, mmax=mmax, nthreads=nthreads, iter=iter, rtol=rtol)
            func = ducc0.sht.pseudo_analysis if iter else ducc0.sht.adjoint_synthesis
            alms = np.zeros((nmaps, hp.Alm.getsize(lmax=kw["lmax"], mmax=kw["mmax"])), dtype=ctype)
            func(map=maps, spin=np.abs(spin), alm=alms, **kw, **kwargs)
            if not iter:
                alms *= self.pixelarea
            return list(alms)

    def alm2map_spin(self, alms, spin, lmax=None, mmax=None, *, nthreads=None, **kwargs):
        if spin ==0:
            return list(-self.alm2map(alms, lmax=lmax, mmax=mmax, pol=False, nthreads=nthreads, **kwargs))
        else:
            alms = np.atleast_2d(alms)
            nmaps = alms.shape[0]
            assert nmaps == 2, "spin function only accepts 2 maps"
            ctype = alms.dtype
            assert ctype in [np.complex64, np.complex128]
            dtype = np.float64 if ctype == np.complex128 else np.float32

            maps = np.zeros((nmaps, hp.nside2npix(self.nside)), dtype=dtype)
            func = ducc0.sht.synthesis
            if lmax is None:
                lmax = hp.Alm.getlmax(alms.shape[-1])
            kw = self.get_kwargs(lmax=lmax, mmax=mmax, nthreads=nthreads)
            func(map=maps, spin=np.abs(spin), alm=alms, **kw, **kwargs)
            return list(maps)

    def smoothing(self, maps_in, fwhm=0.0, sigma=None, beam_window=None, pol=True, nthreads=None):
        alms = self.map2alm(maps_in, pol=pol, nthreads=nthreads)
        alms = hp.smoothalm(alms, fwhm=fwhm, sigma=sigma, beam_window=beam_window, pol=pol)
        return self.alm2map(alms, pol=pol, nthreads=nthreads)
