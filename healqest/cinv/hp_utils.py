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
