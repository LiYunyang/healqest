import numpy as np
import healpy as hp
import sys
import io
sys.path.insert(0, '../src/')
import weights, qest, resp
import pytest
from itertools import combinations


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
    q2 = qest.qest_plus(config, cls)
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


# @pytest.mark.skip
@pytest.mark.parametrize("est1, est2", combinations(['TT', 'EE', "TE", "TB", "EB", "ET", "BT", "BE"], 2))
def test_cmbpluslens_resp_grad(fake_data, est1, est2):
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    q1 = fake_data['q1']
    q2 = fake_data['q2']
    f1 = fake_data['f1']
    f2 = fake_data['f2']

    r1 = q1.get_aresp(f1, f2, est1, est2, )
    r2 = q2.get_aresp(f1, f2, est1, est2, )

    assert np.allclose(r1, r2) or np.allclose(r1, r2)  # , f"grad result not match for {est1}-{est2}"


# @pytest.mark.skip
@pytest.mark.parametrize("est1, est2", combinations(['TT', 'EE', "TE", "TB", "EB", "ET", "BT", "BE"], 2))
def test_cmbpluslens_resp_curl(fake_data, est1, est2):
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    q1 = fake_data['q1']
    q2 = fake_data['q2']
    f1 = fake_data['f1']
    f2 = fake_data['f2']

    r1 = q1.get_aresp(f1, f2, est1+'curl', est2+'curl', )
    r2 = q2.get_aresp(f1, f2, est1+'curl', est2+'curl', )
    assert np.allclose(r1, r2) or np.allclose(r1, r2) # f"curl result not match for {est1}-{est2}"
