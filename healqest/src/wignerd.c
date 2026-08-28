// routines for wigner d-matrix sums and
// integrals using Gauss-Legendre quadrature.
//
// copied from ist_gauss_legendre_pixelization
// and ist_wignerd.cpp by Kendrick Smith.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <memory.h>
#include <omp.h>

void init_gauss_legendre_quadrature(int n, double *x, double *w)
{
    double  z, z1, p1, p2, p3, pp;
    int     i, l;
    int     m = (n+1)/2;

    assert(n>0);

    for (i = 0; i < m; i++) {
	z = cos( M_PI * ((double)i + 0.75) / ((double)n + 0.5) );     /* initial guess for root */

	do {
	    /* at end of this block, p1=P_l(z) and p2=P_{l-1}(z) */
	    /* note recursion is  (l)P_l = (2l-1)zP_{l-1} - (l-1)P_{l-2} */
	    p1 = 1.0;
	    p2 = 0.0;
	    for (l = 1; l <= n; l++) {
		p3 = p2;
		p2 = p1;
		p1 = ((2*l-1)*z*p2 - (l-1)*p3) / (double)l;
	    }

	    pp = n * (z*p1-p2) / (z*z - 1.0);                                        /* P'_l(z) */
	    z1 = z;
	    z  = z1-p1/pp;                                        /* update z (Newton's method) */
	} while (fabs(z-z1) > 1.0e-15);

	w[i] = w[n-1-i] = 2.0 / ((1-z*z)*pp*pp);
	x[i]     = -z;
	x[n-1-i] =  z;
    }
}

static inline double alpha(int l, int s1, int s2)
{
    int ix, iy;

    if (l <= abs(s1) || l <= abs(s2))
	return 0.0;

    ix = l*l - s1*s1;
    iy = l*l - s2*s2;
    return sqrt((double)ix * (double)iy) / (double)l;
}

// the return value is the initial value of l
static inline int wiginit(int s1, int s2, int ntheta, const double *cos_theta, double *out)
{
    double  s12sign   = ((s1+s2)%2) ? -1.0 : 1.0;
    double  prefactor = 1.0;
    int     abs_s1;
    int     i, swap;

    if (abs(s1) > abs(s2)) {
	prefactor *= s12sign;
	swap = s1;
	s1 = s2;
	s2 = swap;
	//swap(s1,s2);
    }
    if (s2 < 0) {
	prefactor *= s12sign;
	s1 = -s1;
	s2 = -s2;
    }

    abs_s1 = abs(s1);
    assert(abs_s1 <= s2);

    for (i = 1; i <= s2-abs_s1; i++)
	prefactor *= sqrt((double)(s2+abs_s1+i)/(double)i);

    for (i = 0; i < ntheta; i++)
	out[i] = prefactor * pow((1.0+cos_theta[i])/2.0, 0.5 * (double)(s2+s1)) * pow((1.0-cos_theta[i])/2.0, 0.5 * (double)(s2-s1));

    return s2;
}

//
// recursion l -> (l+1)
//
static inline void wigrec(int l, int s1, int s2, int ntheta, const double *cos_theta, double *wigd_hi, double *wigd_lo)
{
    int i; double x;
    double  alpha_hi  = alpha(l+1, s1, s2);
    double  alpha_lo  = alpha(l, s1, s2);
    double  beta      = (s1==0 || s2==0) ? 0.0 : ((double)s1 * (double)s2 / (double)(l*(l+1)));

    for (i = 0; i < ntheta; i++) {
	x = (double)(2*l+1)*(cos_theta[i]-beta)*wigd_hi[i] - alpha_lo*wigd_lo[i];
	wigd_lo[i] = wigd_hi[i];
	wigd_hi[i] = x / alpha_hi;
    }
}

//
// helper for ist_wignerd_cf_from_cl
//
static void cf_accum(int l, int nfunc, int ntheta, int lmax, const double *wigd, double *out_cf, const double *in_cl)
{
    int i, f;
    double cl;

    assert(l <= lmax);

    for (f = 0; f < nfunc; f++) {
	cl = in_cl[f*(lmax+1)+l];
	for (i = 0; i < ntheta; i++)
	    out_cf[f*ntheta+i] += cl * wigd[i];
    }
}


//
// This routine computes
//   C_l -> sum_l C_l d^l_{ss'}(theta)
//
// cos_theta: array of length ntheta
// out_cf:    array of length (nfunc)-by-(ntheta)
// in_cl:     array of length (nfunc)-by-(lmax+1)
//
void wignerd_cf_from_cl(int s1, int s2, int nfunc, int ntheta, int lmax, const double *cos_theta, double *out_cf, const double *in_cl)
{
    int l;
    double *wigd_lo, *wigd_hi;

    wigd_lo = (double *)calloc( ntheta, sizeof(double) );
    wigd_hi = (double *)calloc( ntheta, sizeof(double) );

    memset(out_cf, 0, nfunc * ntheta * sizeof(double));

    l = wiginit(s1, s2, ntheta, cos_theta, &wigd_hi[0]);
    if (l <= lmax)
	cf_accum(l, nfunc, ntheta, lmax, &wigd_hi[0], out_cf, in_cl);

    while (l < lmax) {
	wigrec(l, s1, s2, ntheta, cos_theta, &wigd_hi[0], &wigd_lo[0]);
	l++;
	cf_accum(l, nfunc, ntheta, lmax, &wigd_hi[0], out_cf, in_cl);
    }

    free(wigd_lo); free(wigd_hi);
}

//
// helper for ist_wignerd_cl_from_cf
//
static void cl_fill(int l, int nfunc, int ntheta, int lmax, const double *wigd, double *out_cl, const double *in_cf)
{
    int f, i;
    double cl;

    assert(l <= lmax);

    for (f = 0; f < nfunc; f++) {
	cl = 0.0;
	for (i = 0; i < ntheta; i++)
	    cl += wigd[i] * in_cf[f*ntheta+i];
	out_cl[f*(lmax+1)+l] = cl;
    }
}

//
// This routine computes
//   f(theta) -> sum_theta W(theta) f(theta) d^l_{ss'}(theta)
//
// cos_theta:            array of length ntheta
// integration_weights:  array of length ntheta
// out_cl:               array of length (nfunc)-by-(lmax+1)
// in_cf:                array of length (nfunc)-by-(ntheta)
//
void wignerd_cl_from_cf(int s1, int s2, int nfunc, int ntheta, int lmax, const double *cos_theta, const double *integration_weights, double *out_cl, const double *in_cf)
{
    int i, l;
    double *wigd_lo, *wigd_hi;

    wigd_lo = (double *)calloc( ntheta, sizeof(double) );
    wigd_hi = (double *)calloc( ntheta, sizeof(double) );

    memset(out_cl, 0, nfunc * (lmax+1) * sizeof(double));

    l = wiginit(s1, s2, ntheta, cos_theta, &wigd_hi[0]);

    for (i = 0; i < ntheta; i++)
	wigd_hi[i] *= integration_weights[i];

    if (l <= lmax)
	cl_fill(l, nfunc, ntheta, lmax, &wigd_hi[0], out_cl, in_cf);

    while (l < lmax) {
	wigrec(l, s1, s2, ntheta, cos_theta, &wigd_hi[0], &wigd_lo[0]);
	l++;
	cl_fill(l, nfunc, ntheta, lmax, &wigd_hi[0], out_cl, in_cf);
    }

    free(wigd_lo); free(wigd_hi);
}

#define WIGNERD_OMP_BLOCK_SIZE 256


void init_gauss_legendre_quadrature_omp(int n, double *x, double *w)
{
    int m = (n+1)/2;

    assert(n>0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        double z = cos(M_PI * ((double)i + 0.75) / ((double)n + 0.5));
        double z1, p1, p2, p3, pp;
        int l;

        do {
            p1 = 1.0;
            p2 = 0.0;
            for (l = 1; l <= n; l++) {
                p3 = p2;
                p2 = p1;
                p1 = ((2*l-1)*z*p2 - (l-1)*p3) / (double)l;
            }
            pp = n * (z*p1-p2) / (z*z - 1.0);
            z1 = z;
            z = z1-p1/pp;
        } while (fabs(z-z1) > 1.0e-15);

        w[i] = w[n-1-i] = 2.0 / ((1-z*z)*pp*pp);
        x[i] = -z;
        x[n-1-i] = z;
    }
}


int wignerd_cf_from_cl_omp(int s1, int s2, int nfunc, int ntheta, int lmax,
                            const double *cos_theta, double *out_cf, const double *in_cl)
{
    memset(out_cf, 0, nfunc * ntheta * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int block = 0; block < ntheta; block += WIGNERD_OMP_BLOCK_SIZE) {
        double wigd_lo[WIGNERD_OMP_BLOCK_SIZE] = {0.0};
        double wigd_hi[WIGNERD_OMP_BLOCK_SIZE];
        int block_size = ntheta-block < WIGNERD_OMP_BLOCK_SIZE ? ntheta-block : WIGNERD_OMP_BLOCK_SIZE;
        int f, i;
        int l = wiginit(s1, s2, block_size, cos_theta+block, wigd_hi);

        if (l <= lmax) {
            for (f = 0; f < nfunc; f++) {
                double cl = in_cl[f*(lmax+1)+l];
                #pragma omp simd
                for (i = 0; i < block_size; i++)
                    out_cf[f*ntheta+block+i] += cl * wigd_hi[i];
            }
        }

        while (l < lmax) {
            wigrec(l, s1, s2, block_size, cos_theta+block, wigd_hi, wigd_lo);
            l++;
            for (f = 0; f < nfunc; f++) {
                double cl = in_cl[f*(lmax+1)+l];
                #pragma omp simd
                for (i = 0; i < block_size; i++)
                    out_cf[f*ntheta+block+i] += cl * wigd_hi[i];
            }
        }
    }

    return 0;
}


int wignerd_cl_from_cf_omp(int s1, int s2, int nfunc, int ntheta, int lmax,
                            const double *cos_theta, const double *integration_weights,
                            double *out_cl, const double *in_cf)
{
    double *partial_cl;
    int nthreads;
    size_t nout;

    nout = (size_t)nfunc * (size_t)(lmax+1);
    memset(out_cl, 0, nout * sizeof(double));

    nthreads = omp_get_max_threads();
    partial_cl = (double *)calloc((size_t)nthreads * nout, sizeof(double));
    if (partial_cl == NULL)
        return -1;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *local_cl = partial_cl + (size_t)tid * nout;

        #pragma omp for schedule(static)
        for (int block = 0; block < ntheta; block += WIGNERD_OMP_BLOCK_SIZE) {
            double wigd_lo[WIGNERD_OMP_BLOCK_SIZE] = {0.0};
            double wigd_hi[WIGNERD_OMP_BLOCK_SIZE];
            int block_size = ntheta-block < WIGNERD_OMP_BLOCK_SIZE ? ntheta-block : WIGNERD_OMP_BLOCK_SIZE;
            int f, i;
            int l = wiginit(s1, s2, block_size, cos_theta+block, wigd_hi);

            #pragma omp simd
            for (i = 0; i < block_size; i++)
                wigd_hi[i] *= integration_weights[block+i];

            if (l <= lmax) {
                for (f = 0; f < nfunc; f++) {
                    double cl = 0.0;
                    #pragma omp simd reduction(+:cl)
                    for (i = 0; i < block_size; i++)
                        cl += wigd_hi[i] * in_cf[f*ntheta+block+i];
                    local_cl[f*(lmax+1)+l] += cl;
                }
            }

            while (l < lmax) {
                wigrec(l, s1, s2, block_size, cos_theta+block, wigd_hi, wigd_lo);
                l++;
                for (f = 0; f < nfunc; f++) {
                    double cl = 0.0;
                    #pragma omp simd reduction(+:cl)
                    for (i = 0; i < block_size; i++)
                        cl += wigd_hi[i] * in_cf[f*ntheta+block+i];
                    local_cl[f*(lmax+1)+l] += cl;
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < nout; k++) {
        double value = 0.0;
        int t;
        for (t = 0; t < nthreads; t++)
            value += partial_cl[(size_t)t*nout+k];
        out_cl[k] = value;
    }

    free(partial_cl);
    return 0;
}

/*
 * Local variables:
 *  c-basic-offset: 4
 * End:
 */
