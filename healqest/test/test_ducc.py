
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
    g.pixel_weights
    g.ring_weights
    alm = g.map2alm(m, iter=0, pol=True)
    return dict(g=g, m=m, nside=nside, mask=mask, alm=alm)


@pytest.mark.parametrize("pol", [False, True])
@pytest.mark.parametrize("weights", [None, 'pixel', 'ring'])
def test_map2alm(setup, pol, weights):
    if weights is None:
        use_pixel_weights, use_weights = False, False
    elif weights == 'ring':
        use_pixel_weights, use_weights = False, True
    elif weights == 'pixel':
        use_pixel_weights, use_weights = True, False
    else:
        raise
    print(f"\nTesting map2alm: pol={pol}, use_pixel_weights={use_pixel_weights}, use_weights={use_weights}")
    t1 = perf_counter()
    alm1 = hp.map2alm(setup['m'], pol=pol, iter=0, use_pixel_weights=use_pixel_weights, use_weights=use_weights)
    t2 = perf_counter()
    alm2 = setup['g'].map2alm(setup['m'], pol=pol, iter=0, use_pixel_weights=use_pixel_weights, use_weights=use_weights)
    t3 = perf_counter()
    assert np.allclose(alm1, alm2)
    print(f"healpy: {(t2-t1)*1000:.2f}ms; ducc is {(t2-t1)/(t3-t2):.2f}x faster.")


@pytest.mark.parametrize("spin", [0, 1, 2])
def test_map2alm_spin(setup, spin):
    print(f"\nTesting map2alm_spin: spin={spin}")
    m = setup['m'] if spin ==0 else setup['m'][1:]
    t1 = perf_counter()
    alm1 = hp.map2alm_spin(m, spin=spin)
    t2 = perf_counter()
    alm2 = setup['g'].map2alm_spin(m, spin=spin)
    t3 = perf_counter()
    assert np.allclose(alm1, alm2)
    print(f"healpy: {(t2-t1)*1000:.2f}ms; ducc is {(t2-t1)/(t3-t2):.2f}x faster.")


@pytest.mark.parametrize("spin", [0, 1, 2])
def test_map2alm_spin(setup, spin):
    print(f"\nTesting map2alm_spin: spin={spin}")
    m = setup['m'] if spin ==0 else setup['m'][1:]
    t1 = perf_counter()
    alm1 = hp.map2alm_spin(m, spin=spin)
    t2 = perf_counter()
    alm2 = setup['g'].map2alm_spin(m, spin=spin)
    t3 = perf_counter()
    assert np.allclose(alm1, alm2)
    print(f"healpy: {(t2-t1)*1000:.2f}ms; ducc is {(t2-t1)/(t3-t2):.2f}x faster.")


@pytest.mark.parametrize("pol", [False, True])
def test_alm2map(setup, pol):
    print(f"\nTesting alm2map: pol={pol}")
    alm = setup['alm']

    t1 = perf_counter()
    m1 = hp.alm2map(alm, nside=setup['nside'], pol=pol)
    t2 = perf_counter()
    m2 = setup['g'].alm2map(alm, pol=pol)
    t3 = perf_counter()
    assert np.allclose(m1*setup['mask'], m2*setup['mask'])
    print(f"healpy: {(t2-t1)*1000:.2f}ms; ducc is {(t2-t1)/(t3-t2):.2f}x faster.")


@pytest.mark.parametrize("spin", [0, 1, 2])
def test_alm2map_spin(setup, spin):
    print(f"\nTesting alm2map_spin: spin={spin}")
    alm = setup['alm'] if spin ==0 else setup['alm'][1:]
    lmax = hp.Alm.getlmax(len(alm[0]))
    t1 = perf_counter()
    m1 = hp.alm2map_spin(alm, nside=setup['nside'], spin=spin, lmax=lmax)
    t2 = perf_counter()
    m2 = setup['g'].alm2map_spin(alm, spin=spin)
    t3 = perf_counter()
    assert np.allclose(m1*setup['mask'], m2*setup['mask'])
    print(f"healpy: {(t2-t1)*1000:.2f}ms; ducc is {(t2-t1)/(t3-t2):.2f}x faster.")


"""
# to run the test:
>>> pytest -vs test_ducc.py 
"""