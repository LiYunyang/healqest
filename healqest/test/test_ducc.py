
import sys
sys.path.insert(0, '../src/')
import ducc_sht
import numpy as np
import healpy as hp
import pytest
from time import perf_counter


@pytest.fixture
def setup():
    nside = 128
    dec_range = [-70, -40]
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    mask = np.logical_and(dec > dec_range[0], dec < dec_range[1])
    g = ducc_sht.Geometry(nside=nside, dec_range=[dec_range[0], dec_range[1]])
    m = np.random.normal(0, 1, (3, hp.nside2npix(nside)))*mask
    # pre-loading the cache
    g.pixel_weights
    g.ring_weights
    alm = g.map2alm(m, iter=0, pol=True)
    return dict(g=g, m=m, nside=nside, mask=mask, alm=alm)


def print_time(t1, t2, t3):
    print(f"healpy: {(t2-t1)*1000:.2f}ms; ducc is\033[32m {(t2-t1)/(t3-t2):.2f}x \033[0mfaster.")


@pytest.mark.parametrize("check_bad", [False, True])
@pytest.mark.parametrize("pol", [False, True])
@pytest.mark.parametrize("weights", [None, 'pixel', 'ring'])
def test_map2alm(setup, pol, weights, check_bad):
    if weights is None:
        use_pixel_weights, use_weights = False, False
    elif weights == 'ring':
        use_pixel_weights, use_weights = False, True
    elif weights == 'pixel':
        use_pixel_weights, use_weights = True, False
    else:
        raise
    print(f"\nTesting map2alm: pol={pol}, use_pixel_weights={use_pixel_weights}, use_weights={use_weights},"
          f"check_bad={check_bad}")
    m = setup['m'].copy()
    if check_bad:
        m[:, ~setup['mask']] = hp.UNSEEN
    m0 = m.copy()
    t1 = perf_counter()
    alm1 = hp.map2alm(m, pol=pol, iter=0, use_pixel_weights=use_pixel_weights, use_weights=use_weights)
    t2 = perf_counter()
    alm2 = setup['g'].map2alm(m, pol=pol, iter=0, use_pixel_weights=use_pixel_weights, use_weights=use_weights,
                              check=check_bad)
    t3 = perf_counter()
    assert np.allclose(alm1, alm2)
    assert np.all(m==m0), "data mutation!"
    print_time(t1, t2, t3)


@pytest.mark.parametrize("check_bad", [False, True])
@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_map2alm_spin(setup, spin, check_bad):
    print(f"\nTesting map2alm_spin: spin={spin}, check_bad={check_bad}")
    m = setup['m'].copy() if spin ==0 else setup['m'][1:].copy()
    if check_bad:
        m[:, ~setup['mask']] = hp.UNSEEN
    m0 = m.copy()
    t1 = perf_counter()
    alm1 = hp.map2alm_spin(m, spin=spin)
    t2 = perf_counter()
    alm2 = setup['g'].map2alm_spin(m, spin=spin, check=check_bad)
    t3 = perf_counter()
    assert np.allclose(alm1, alm2)
    assert np.all(m==m0), "data mutation!"
    print_time(t1, t2, t3)


@pytest.mark.parametrize("pol", [False, True])
@pytest.mark.parametrize("buffer", [False, True])
def test_alm2map(setup, pol, buffer):
    print(f"\nTesting alm2map: pol={pol}, w/ or w/o buffer={buffer}")
    alm = setup['alm']
    alm0 = alm.copy()
    t1 = perf_counter()
    m1 = hp.alm2map(alm, nside=setup['nside'], pol=pol)
    if buffer:
        m2 = np.zeros_like(m1)
        t2 = perf_counter()
        setup['g'].alm2map(alm, pol=pol, maps=m2)
    else:
        t2 = perf_counter()
        m2 = setup['g'].alm2map(alm, pol=pol)
    t3 = perf_counter()
    assert np.allclose(m1*setup['mask'], m2*setup['mask'])
    assert np.all(alm==alm0), "data mutation!"
    print_time(t1, t2, t3)


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
@pytest.mark.parametrize("buffer", [False, True])
def test_alm2map_spin(setup, spin, buffer):
    print(f"\nTesting alm2map_spin: spin={spin}, w/ or w/o buffer={buffer}")
    alm = setup['alm'] if spin ==0 else setup['alm'][1:]
    alm0 = alm.copy()
    lmax = hp.Alm.getlmax(len(alm[0]))
    t1 = perf_counter()
    m1 = hp.alm2map_spin(alm, nside=setup['nside'], spin=spin, lmax=lmax)
    if buffer:
        m2 = np.zeros_like(m1)
        t2 = perf_counter()
        setup['g'].alm2map_spin(alm, spin=spin, maps=m2)
    else:
        t2 = perf_counter()
        m2 = setup['g'].alm2map_spin(alm, spin=spin)
    t3 = perf_counter()
    assert np.allclose(m1*setup['mask'], m2*setup['mask'])
    assert np.all(alm == alm0), "data mutation!"
    print_time(t1, t2, t3)


@pytest.mark.parametrize("fwhm", [2])
@pytest.mark.parametrize("pol", [True, False])
@pytest.mark.parametrize("weights", [None, 'pixel', 'ring'])
def test_smoothing(setup, fwhm, pol, weights):
    if weights is None:
        use_pixel_weights, use_weights = False, False
    elif weights == 'ring':
        use_pixel_weights, use_weights = False, True
    elif weights == 'pixel':
        use_pixel_weights, use_weights = True, False
    else:
        raise
    print(f"\nTesting smoothing: fwhm={fwhm}, pol={pol}, use_pixel_weights={use_pixel_weights}, "
          f"use_weights={use_weights}")

    m = setup['m']
    m[:, setup['mask']] = hp.UNSEEN
    t1 = perf_counter()
    m1 = hp.smoothing(m, fwhm=np.deg2rad(fwhm), pol=pol, iter=0, use_weights=use_weights,
                      use_pixel_weights=use_pixel_weights)
    t2 = perf_counter()
    m2 = setup['g'].smoothing(m, fwhm=np.deg2rad(fwhm), pol=pol, iter=0, use_weights=use_weights,
                              use_pixel_weights=use_pixel_weights)

    t3 = perf_counter()
    assert np.allclose(m1*setup['mask'], m2*setup['mask'])
    print_time(t1, t2, t3)


"""
# to run the test:
>>> pytest -vs test_ducc.py 
"""