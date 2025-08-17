import time

import numpy as np
import healpy as hp

import pytest
from itertools import product
from healqest import weights, qest, resp


@pytest.fixture(autouse=True, scope="module")
def apply_custom_pytest_options(request):
    """
    Apply custom pytest options (-vs --tb=no) for this module.
    """
    # Enable verbose mode (-v)
    request.config.option.verbose = 2
    # Disable output capturing (-s)
    request.config.option.capture = "no"
    # Disable traceback (--tb=no)
    request.config.option.tbstyle = "no"


def print_time(t1, t2, t3):
    print(f"std: {(t2-t1)*1000:.2f}ms; fast is\033[32m {(t2-t1)/(t3-t2):.2f}x \033[0mfaster.")


@pytest.fixture(scope="module")
def fake_data():
    nside= 16
    lmax = 30
    np.random.seed(37)
    alm1 = hp.map2alm(np.random.normal(0, 1, (3, hp.nside2npix(nside))), lmax=lmax)
    alm2 = hp.map2alm(np.random.normal(0, 1, (3, hp.nside2npix(nside))), lmax=lmax)
    ell = np.arange(lmax + 1)
    cls = dict(
        ucmb=dict(
            tt=ell * (ell + 1),
            ee=ell * (ell + 1) / 10,
            te=np.sin(ell / 5) / 2,
            bb=np.ones_like(ell)*0  # qest disables "bb" terms, so the qe test won't pass
        )
    )
    f1 = 1e-6 * ell ** 0.5
    f2 = 1e-6 * ell ** 2
    estimator = qest.Qest(lmax=lmax, nside=nside, Cls=cls['ucmb'], Lmax=lmax)
    return dict(estimator=estimator, alm1=alm1, alm2=alm2, f1=f1, f2=f2)


# @pytest.mark.skip
@pytest.mark.parametrize("est1, est2", product(['TT', 'EE', "TE", "TB", "EB", "ET", "BT", "BE"], repeat=2))
@pytest.mark.parametrize("curl", [False, True])
def test_fast_resp(fake_data, est1, est2, curl):
    f1 = fake_data['f1']
    f2 = fake_data['f2']
    estimator = fake_data['estimator']
    t1 = time.perf_counter()
    r1 = estimator.get_aresp(f1, f2, qe1=est1, qe2=est2, curl=curl)
    t2 = time.perf_counter()
    r2 = estimator.get_aresp(f1, f2, qe1=est1, qe2=est2, curl=curl, fast=True)
    t3 = time.perf_counter()

    assert np.allclose(r1.real, r2.real)
    print_time(t1, t2, t3)


@pytest.mark.parametrize("qe1, qe2", product(['TT', 'EE', "TE", "TB", "EB", "ET", "BT", "BE"], repeat=2))
@pytest.mark.parametrize("curl", [False, True])
def test_fast_san0(fake_data, qe1, qe2, curl):

    estimator = fake_data['estimator']

    almbars1 = fake_data['alm1']
    almbars2 = fake_data['alm2']

    qeXY = weights.weights_plus(qe1 if not curl else qe1+'curl', estimator.cls, estimator.lmax)
    qeZA = weights.weights_plus(qe2 if not curl else qe2+'curl', estimator.cls, estimator.lmax)

    i1, i2 = 'TEB'.index(qe1[0]), 'TEB'.index(qe1[1])
    j1, j2 = 'TEB'.index(qe2[0]), 'TEB'.index(qe2[1])

    XZ = hp.alm2cl(almbars1[i1], almbars2[j1])
    YA = hp.alm2cl(almbars1[i2], almbars2[j2])
    XA = hp.alm2cl(almbars1[i1], almbars2[j2])
    YZ = hp.alm2cl(almbars1[i2], almbars2[j1])

    ret = np.zeros(estimator.Lmax + 1, dtype=np.complex128)
    t1 = time.perf_counter()
    r1= resp.fill_clq1q2_fullsky(qeXY, qeZA, ret, XZ, YA, XA, YZ)
    t2 = time.perf_counter()
    ret = np.zeros(estimator.Lmax + 1, dtype=np.complex128)
    r2 = resp.fill_clq1q2_fullsky(qeXY, qeZA, ret, XZ, YA, XA, YZ, fast=True)
    t3 = time.perf_counter()

    assert np.allclose(r1.real, r2.real)
    print_time(t1, t2, t3)
