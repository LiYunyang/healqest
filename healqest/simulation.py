"""Simulation helpers."""

import healpy as hp
import numpy as np

from . import healqest_utils as hq
from .startup import Config


def sample_joint_alms(cls, seed=None):
    """Sample scalar-real-sky ALMs from ``cls[i, j, l]``."""
    cls = np.moveaxis(cls, -1, 0)
    rng = np.random.default_rng(seed)
    lmax, nfield = cls.shape[0] - 1, cls.shape[1]
    ell, m = hp.Alm.getlm(lmax)
    eigenvalues, eigenvectors = np.linalg.eigh(cls)
    factors = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))[..., None, :]
    modes = (rng.normal(size=(nfield, len(ell))) + 1j * rng.normal(size=(nfield, len(ell)))) / np.sqrt(2)
    modes[:, m == 0] = rng.normal(size=(nfield, np.count_nonzero(m == 0)))
    return np.einsum("aik,ka->ia", factors[ell], modes, optimize=True)


def sample_agora_alms(fname_cls, fname_alm, cond_comp=(), seed=None):
    """Return ``(alm_90, alm_150, alm_220)`` for unconditioned AGORA components."""
    COMPONENTS = ("rad", "cib", "tsz")
    FREQUENCIES = (90, 150, 220)
    CHANNELS = [(component, freq) for component in COMPONENTS for freq in FREQUENCIES]

    cls = np.load(Config.path(fname_cls))
    cond_comp = tuple(cond_comp)
    assert set(cond_comp) <= set(COMPONENTS)
    lmax = cls.shape[-1] - 1
    observed = [i for i, (component, _) in enumerate(CHANNELS) if component in cond_comp]
    unobserved = [i for i in range(len(CHANNELS)) if i not in observed]
    ell, m = hp.Alm.getlm(lmax)

    if not observed:
        alms = sample_joint_alms(cls, seed)
    elif not unobserved:
        alms = np.zeros((0, len(ell)), dtype=complex)
    else:
        cls = np.moveaxis(cls, -1, 0)
        observed_alms = []
        for i in observed:
            component, freq = CHANNELS[i]
            alm = hp.read_alm(Config.path(fname_alm, freq=freq, comp=component))
            alm_lmax = hp.Alm.getlmax(alm.size)
            assert alm_lmax >= lmax
            observed_alms.append(hq.reduce_lmax(alm, lmax) if alm_lmax > lmax else alm)
        observed_alms = np.array(observed_alms)

        cov_oo = cls[:, observed][:, :, observed]
        cov_uo = cls[:, unobserved][:, :, observed]
        gain = cov_uo @ np.linalg.pinv(cov_oo)
        covariance = cls[:, unobserved][:, :, unobserved] - gain @ np.swapaxes(cov_uo, -1, -2)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        factors = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))[..., None, :]
        rng = np.random.default_rng(seed)
        modes = (
            rng.normal(size=(len(unobserved), len(ell))) + 1j * rng.normal(size=(len(unobserved), len(ell)))
        ) / np.sqrt(2)
        modes[:, m == 0] = rng.normal(size=(len(unobserved), np.count_nonzero(m == 0)))
        mean = np.einsum("aio,oa->ia", gain[ell], observed_alms, optimize=True)
        alms = mean + np.einsum("aik,ka->ia", factors[ell], modes, optimize=True)

    return tuple(
        alms[[j for j, i in enumerate(unobserved) if CHANNELS[i][1] == freq]].sum(axis=0)
        for freq in FREQUENCIES
    )


def scale_agora_maps(fname_map, Asz, Arad, Acib):
    """Return scaled ``(map_90, map_150, map_220)``."""
    amplitudes = {"tsz": Asz, "rad": Arad, "cib": Acib}
    return tuple(
        sum(
            amplitude * hq.read_map(Config.path(fname_map, freq=freq, comp=component))
            for component, amplitude in amplitudes.items()
        )
        for freq in (90, 150, 220)
    )
