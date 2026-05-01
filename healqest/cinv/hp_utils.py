"""Convinient functions for curved-sky lensing reconstruction."""

import healpy as hp
import numpy as np
from .. import log

logger = log.get_logger(__name__)


def cl2almformat(cl):
    """Repeat Cl for all m-modes at each ell.

    return alm-ordering array cl array starts with ell=0
    """
    lmax = len(cl) - 1
    alm = np.zeros(hp.Alm.getsize(lmax))
    idx = 0
    for i in range(0, lmax + 1):
        alm[idx : idx + (lmax + 1 - i)] = cl[i:]
        idx = idx + (lmax + 1 - i)
    return alm


def read_map(m):
    """Reads a map whether given as (list of) string (with ',f' denoting field f), array or callable."""
    if callable(m):
        return m()
    if isinstance(m, list):
        ma = read_map(m[0])
        for m2 in m[1:]:
            ma *= read_map(m2)
        return ma
    if not isinstance(m, str):
        return m
    if "," not in m:
        return hp.read_map(m)
    m, field = m.split(",")
    return hp.read_map(m, field=int(field))


class jit:
    """just-in-time instantiation wrapper class."""

    def __init__(self, ctype, *cargs, **ckwds):
        self.__dict__["__jit_args"] = [ctype, cargs, ckwds]
        self.__dict__["__jit_obj"] = None

    def instantiate(self):
        [ctype, cargs, ckwds] = self.__dict__["__jit_args"]
        logger.info(f"jit: instantiating ctype={ctype}")
        self.__dict__["__jit_obj"] = ctype(*cargs, **ckwds)
        del self.__dict__["__jit_args"]

    def __getattr__(self, attr):
        if self.__dict__["__jit_obj"] is None:
            self.instantiate()
        return getattr(self.__dict__["__jit_obj"], attr)

    def __setattr__(self, attr, val):
        if self.__dict__["__jit_obj"] is None:
            self.instantiate()
        setattr(self.__dict__["__jit_obj"], attr, val)
