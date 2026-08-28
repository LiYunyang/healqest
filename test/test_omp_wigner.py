import time

import numpy as np
import pytest

from healqest import resp, weights, wignerd


@pytest.mark.parametrize("npoints", [40, 41])
def test_omp_gauss_legendre_quadrature(npoints):
    serial = wignerd.GaussLegendreQuadrature(npoints)
    omp = wignerd.GaussLegendreQuadrature(npoints, omp=True)

    np.testing.assert_allclose(omp.zvec, serial.zvec, rtol=0, atol=0)
    np.testing.assert_allclose(omp.wvec, serial.wvec, rtol=0, atol=0)


@pytest.mark.parametrize("spins", [(0, 0), (2, 2), (2, -2), (1, -1), (3, 1)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_omp_wignerd_transforms(spins, complex_input):
    rng = np.random.default_rng(37)
    lmax = 30
    npoints = 46
    serial = wignerd.GaussLegendreQuadrature(npoints)
    omp = wignerd.GaussLegendreQuadrature(npoints, omp=True)

    cl = rng.normal(size=lmax + 1)
    cf = rng.normal(size=npoints)
    if complex_input:
        cl = cl + 1j * rng.normal(size=lmax + 1)
        cf = cf + 1j * rng.normal(size=npoints)

    serial_cf = serial.cf_from_cl(*spins, cl)
    omp_cf = omp.cf_from_cl(*spins, cl)
    np.testing.assert_allclose(omp_cf, serial_cf, rtol=1e-12, atol=1e-12)

    serial_cl = serial.cl_from_cf(lmax, *spins, cf)
    omp_cl = omp.cl_from_cf(lmax, *spins, cf)
    np.testing.assert_allclose(omp_cl, serial_cl, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("qe1,qe2", [("TT", "TT"), ("TE", "ET"), ("EB", "BE")])
@pytest.mark.parametrize("curl", [False, True])
@pytest.mark.parametrize("fast", [False, True])
def test_omp_clq1q2_response(qe1, qe2, curl, fast):
    lmax = 20
    ell = np.arange(lmax + 1, dtype=float)
    cls = {
        "tt": 1 + 0.1 * ell,
        "ee": 0.7 + 0.08 * ell,
        "bb": np.zeros_like(ell),
        "te": 0.15 * np.cos(ell / 3),
    }
    qeXY = weights.WeightsPlus(qe1, cls, lmax, curl=curl)
    qeZA = weights.WeightsPlus(qe2, cls, lmax, curl=curl)
    spectra = [1 / (ell + offset) for offset in (1, 2, 3, 4)]

    serial = resp.fill_clq1q2_fullsky(qeXY, qeZA, np.zeros(lmax + 1, dtype=complex), *spectra, fast=fast)
    omp = resp.fill_clq1q2_fullsky(
        qeXY, qeZA, np.zeros(lmax + 1, dtype=complex), *spectra, fast=fast, omp=True
    )

    np.testing.assert_allclose(omp, serial, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("curl", [False, True])
def test_omp_fullsky_response(curl):
    lmax = 20
    ell = np.arange(lmax + 1, dtype=float)
    cls = {
        "tt": 1 + 0.1 * ell,
        "ee": 0.7 + 0.08 * ell,
        "bb": np.zeros_like(ell),
        "te": 0.15 * np.cos(ell / 3),
    }
    qeXY = weights.WeightsPlus("TE", cls, lmax, curl=curl)
    qeZA = weights.WeightsPlus("ET", cls, lmax, curl=curl)
    fX = 1 / (ell + 1)
    fY = 1 / (ell + 2)

    serial = resp.fill_resp_fullsky(qeXY, qeZA, np.zeros(lmax + 1, dtype=complex), fX, fY)
    omp = resp.fill_resp_fullsky(qeXY, qeZA, np.zeros(lmax + 1, dtype=complex), fX, fY, omp=True)

    np.testing.assert_allclose(omp, serial, rtol=1e-11, atol=1e-12)


@pytest.mark.benchonly
def test_omp_wignerd_benchmark():
    lmax = 3000
    npoints = 4501
    cl = np.ones(lmax + 1, dtype=complex)
    serial = wignerd.GaussLegendreQuadrature(npoints)
    omp = wignerd.GaussLegendreQuadrature(npoints, omp=True)

    start = time.perf_counter()
    serial.cf_from_cl(0, 0, cl)
    serial_time = time.perf_counter() - start

    start = time.perf_counter()
    omp.cf_from_cl(0, 0, cl)
    omp_time = time.perf_counter() - start

    print(f"serial: {serial_time:.3f}s; OpenMP: {omp_time:.3f}s; speedup: {serial_time / omp_time:.2f}x")
