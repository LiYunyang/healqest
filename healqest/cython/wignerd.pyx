# distutils: language = c
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

    int _wignerd_cl_from_cf_omp "wignerd_cl_from_cf_omp"(
        int s1, int s2, int nfunc, int ntheta, int lmax,
        const double*cos_theta, const double*integration_weights,
        double*out_cl, const double*in_cf
    ) noexcept nogil

    int _wignerd_cf_from_cl_omp "wignerd_cf_from_cl_omp"(
        int s1, int s2, int nfunc, int ntheta, int lmax,
        const double*cos_theta, double*out_cf, const double*in_cl
    ) noexcept nogil

    void _init_gauss_legendre_quadrature_omp "init_gauss_legendre_quadrature_omp"(
        int n, double*x, double*w
    ) noexcept nogil


def wignerd_cl_from_cf(int s1, int s2, int nfunc, int ntheta, int lmax,
                    np.ndarray[np.double_t, ndim=1] cos_theta,
                    np.ndarray[np.double_t, ndim=1] integration_weights,
                    np.ndarray[np.double_t, ndim=1] in_cf, bint omp=False):
    """Direct C function call with array validation"""
    cdef np.ndarray[np.double_t, ndim=1] out_cl = np.empty((lmax + 1)*nfunc, dtype=np.float64)
    cdef const double* cos_theta_ptr = <const double*>cos_theta.data
    cdef const double* integration_weights_ptr = <const double*>integration_weights.data
    cdef const double* in_cf_ptr = <const double*>in_cf.data
    cdef double* out_cl_ptr = <double*>out_cl.data
    cdef int status
    if omp:
        with nogil:
            status = _wignerd_cl_from_cf_omp(
                s1, s2, nfunc, ntheta, lmax,
                cos_theta_ptr, integration_weights_ptr, out_cl_ptr, in_cf_ptr
            )
        if status != 0:
            raise MemoryError("Unable to allocate OpenMP Wigner-d inverse-transform workspace")
    else:
        _wignerd_cl_from_cf(
            s1, s2, nfunc, ntheta, lmax,
            cos_theta_ptr, integration_weights_ptr, out_cl_ptr, in_cf_ptr
        )
    return out_cl

def wignerd_cf_from_cl(int s1, int s2, int nfunc, int ntheta, int lmax,
                    np.ndarray[np.double_t, ndim=1] cos_theta,
                    np.ndarray[np.double_t, ndim=1] in_cl, bint omp=False):
    """Direct C function call with array validation"""
    cdef np.ndarray[np.double_t, ndim=1] out_cf = np.empty(ntheta*nfunc, dtype=np.float64)
    cdef const double* cos_theta_ptr = <const double*>cos_theta.data
    cdef const double* in_cl_ptr = <const double*>in_cl.data
    cdef double* out_cf_ptr = <double*>out_cf.data
    cdef int status
    if omp:
        with nogil:
            status = _wignerd_cf_from_cl_omp(
                s1, s2, nfunc, ntheta, lmax, cos_theta_ptr, out_cf_ptr, in_cl_ptr
            )
        if status != 0:
            raise MemoryError("Unable to allocate OpenMP Wigner-d forward-transform workspace")
    else:
        _wignerd_cf_from_cl(s1, s2, nfunc, ntheta, lmax, cos_theta_ptr, out_cf_ptr, in_cl_ptr)
    return out_cf

def init_gauss_legendre_quadrature(int n, bint omp=False):
    """Initialize quadrature (returns two flat arrays)"""
    cdef np.ndarray[np.double_t, ndim=1] x = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=1] w = np.empty(n, dtype=np.float64)
    cdef double* x_ptr = <double*>x.data
    cdef double* w_ptr = <double*>w.data
    if omp:
        with nogil:
            _init_gauss_legendre_quadrature_omp(n, x_ptr, w_ptr)
    else:
        _init_gauss_legendre_quadrature(n, x_ptr, w_ptr)
    return x, w
