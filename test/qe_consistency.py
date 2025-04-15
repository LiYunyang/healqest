import numpy as np
import healpy as hp
import sys
import io

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


@pytest.fixture(scope="module")
def fake_data():
    nside= 16
    lmax = 30
    np.random.seed(37)
    alm1 = hp.map2alm(np.random.normal(0, 1, hp.nside2npix(nside)), lmax=lmax)
    alm2 = hp.map2alm(np.random.normal(0, 1, hp.nside2npix(nside)), lmax=lmax)
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

    config = dict(lensrec=dict(lmaxT=lmax, lmaxP=lmax, Lmax=lmax, cltype='ucmb', nside=nside))
    q1 = qest.qest(config, cls)
    q2 = qest.Qest(Cls=cls['ucmb'], lmax=lmax, Lmax=lmax, nside=nside)

    return dict(q1=q1, q2=q2, alm1=alm1, alm2=alm2, f1=f1, f2=f2, cls=cls)


@pytest.mark.parametrize("estimator", ['TT', 'EE', "TE", "TB", "EB", "ET", "BT", "BE"])
def test_cmbpluslens_qest(fake_data, estimator):
    q1 = fake_data['q1']
    q2 = fake_data['q2']
    alm1 = fake_data['alm1']
    alm2 = fake_data['alm2']
    glm1, clm1 = q1.eval(estimator, alm1, alm2)
    glm2, clm2 = q2.eval(estimator, alm1, alm2)
    clm3, glm3 = q2.eval(f"{estimator}curl", alm1, alm2)

    assert np.allclose(glm1, glm2), f"grad result not match for {estimator}"
    assert np.allclose(clm1, clm2), f"curl result not match for {estimator}"
    assert np.allclose(clm3, clm2), f"{estimator}curl not match curl of {estimator}"


def two_round_test(r1, r2, message):
    try:
        assert np.allclose(r1, r2)
    except AssertionError as e:
        assert np.allclose(r1, -r2), f"{message} not match, even with a sign flip!"
        pytest.fail(f"{message} not match, but are consistent with a sign flip!")


# @pytest.mark.skip
@pytest.mark.parametrize("est1, est2", product(['TT', 'EE', "TE", "TB", "EB", "ET", "BT", "BE"], repeat=2))
def test_cmbpluslens_resp_grad(fake_data, est1, est2):
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    q1 = fake_data['q1']
    q2 = fake_data['q2']
    f1 = fake_data['f1']
    f2 = fake_data['f2']

    r1 = q1.get_aresp(f1, f2, est1, est2, )
    r2 = q2.get_aresp(f1, f2, est1, est2, )

    if ''.join([est1, est2]) in ['TBEB','TBBE', 'BTEB','BTBE','EBTB','EBBT','BETB','BEBT']:
        r1 *= -1

    two_round_test(r1, r2, f"grad resp result for {est1}-{est2}")


# @pytest.mark.skip
@pytest.mark.parametrize("est1, est2", product(['TT', 'EE', "TE", "TB", "EB", "ET", "BT", "BE"], repeat=2))
def test_cmbpluslens_resp_curl(fake_data, est1, est2):
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    q1 = fake_data['q1']
    q2 = fake_data['q2']
    f1 = fake_data['f1']
    f2 = fake_data['f2']

    r1 = q1.get_aresp(f1, f2, est1+'curl', est2+'curl', )
    r2 = q2.get_aresp(f1, f2, est1+'curl', est2+'curl', )
    if ''.join([est1, est2]) in ['TBEB','TBBE', 'BTEB','BTBE','EBTB','EBBT','BETB','BEBT']:
        pass
        r1 *= -1

    two_round_test(r1, r2, f"curl resp result for {est1}-{est2}")
