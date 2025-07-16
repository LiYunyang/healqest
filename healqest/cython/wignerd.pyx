# distutils: language = c
# distutils: sources = healqest/src/wignerd.c
# cython: cdivision=True


import numpy as np
cimport numpy as np

# Exact C interface from wignerd.h
cdef extern from "wignerd.h":
    void _wignerd_cl_from_cf "wignerd_cl_from_cf"(int s1, int s2, int nfunc, int ntheta, int lmax,
                                                  const double*cos_theta, const double*integration_weights,
                                                  double*out_cl, const double*in_cf)

    void _wignerd_cf_from_cl "wignerd_cf_from_cl"(int s1, int s2, int nfunc, int ntheta, int lmax,
                                                  const double*cos_theta, double*out_cf, const double*in_cl)

    void _init_gauss_legendre_quadrature "init_gauss_legendre_quadrature"(int n, double*x, double*w)


def wignerd_cl_from_cf(int s1, int s2, int nfunc, int ntheta, int lmax,
                    np.ndarray[np.double_t, ndim=1] cos_theta,
                    np.ndarray[np.double_t, ndim=1] integration_weights,
                    np.ndarray[np.double_t, ndim=1] in_cf):
    """Direct C function call with array validation"""
    cdef np.ndarray[np.double_t, ndim=1] out_cl = np.empty((lmax + 1)*nfunc, dtype=np.float64)
    _wignerd_cl_from_cf(s1, s2, nfunc, ntheta, lmax,
                       <const double*>cos_theta.data,
                       <const double*>integration_weights.data,
                       <double*>out_cl.data,
                       <const double*>in_cf.data)
    return out_cl

def wignerd_cf_from_cl(int s1, int s2, int nfunc, int ntheta, int lmax,
                    np.ndarray[np.double_t, ndim=1] cos_theta,
                    np.ndarray[np.double_t, ndim=1] in_cl):
    """Direct C function call with array validation"""
    cdef np.ndarray[np.double_t, ndim=1] out_cf = np.empty(ntheta*nfunc, dtype=np.float64)
    _wignerd_cf_from_cl(s1, s2, nfunc, ntheta, lmax,
                       <const double*>cos_theta.data,
                       <double*>out_cf.data,
                       <const double*>in_cl.data)
    return out_cf

def init_gauss_legendre_quadrature(int n):
    """Initialize quadrature (returns two flat arrays)"""
    cdef np.ndarray[np.double_t, ndim=1] x = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=1] w = np.empty(n, dtype=np.float64)
    _init_gauss_legendre_quadrature(n, <double*>x.data, <double*>w.data)
    return x, w