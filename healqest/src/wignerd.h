#ifndef WIGNERD_H
#define WIGNERD_H

#ifdef __cplusplus
extern "C" {
#endif

void wignerd_cl_from_cf(int s1, int s2, int nfunc, int ntheta, int lmax,
                        const double *cos_theta, const double *integration_weights,
                        double *out_cl, const double *in_cf);

void wignerd_cf_from_cl(int s1, int s2, int nfunc, int ntheta, int lmax,
                        const double *cos_theta,
                        double *out_cf, const double *in_cl);

void init_gauss_legendre_quadrature(int n, double *x, double *w);

#ifdef __cplusplus
}
#endif

#endif