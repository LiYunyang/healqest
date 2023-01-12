import numpy as np
import wignerd

def fill_resp_fullsky(qeXY, qeZA, ret, fX, fY):
    """ compute the response of this estimator to the statistical
    anisotropy encapsulated by a second estimator qeZA,
        
        R(L) = 1/2 \int{d^2 l_X} \int{d_2 l_Y}
                         W^{XY} W^{ZA} fX(l_X) fY(l_Y).
    with l_X+l_Y=L and fX(l_X) fY(l_y) represent filters which are
    diagonal in Fourier space applied to the X and Y fields.
    dividing the output of self.eval() by this response gives a properly
    normalized estimator for the statistical anisotropy defined by qeZA.
    """
    ret[:] = 0.0
    qe_cov_fill_helper_fullsky(qeXY, qeZA, ret, fX, fY, switch_ZA=False, conj_ZA=False)
    ret[:] *= 2.0 # multiply by 2 because qe_cov_fill_helper returns 1/2 the response.
    return ret.real

    
def qe_cov_fill_helper_fullsky( qeXY, qeZA, ret, fX, fY, switch_ZA=False, conj_ZA=False):
    """ a full-sky version of qe_cov_fill_helper_flatsky.
    
         * qeXY      = first estimator.
         * qeZA      = second estimator.
         * ret       = complex array in which to store results (the length of this array, lmax+1, defines the maximum multipole).
         * fX, fY    = 1D real arrays representing the filter functions for the X and Y fields.
         * switch_ZA = change W_{ZA}^{j}(l_X, l_Y, L) -> W_{ZA}^{j, ZA}(l_Y, l_X, L) .
         * conj_ZA   = take the complex conjugate of the W_{ZA} weight function. W_{ZA}^{j} -> W^{ZA}^{* j}.
    """

    lmax = len(ret)-1
    
    i1_ZA, i2_ZA = { False : (0,1), True : (1,0) }[switch_ZA]
    cfunc_ZA     = { False : lambda v : v, True : lambda v : np.conj(v) }[conj_ZA]

    lmax_fX      = len(fX)-1
    lmax_fY      = len(fY)-1
    
    for i in range(0, qeXY.ntrm):
        for j in range(0, qeZA.ntrm):
            # l1 part
            tl1min = max(abs(qeXY.s[i][0]), abs(qeZA.s[j][i1_ZA]))
            tl1max = min( [qeXY.lmax, qeZA.lmax, lmax_fX] )
            
            cl1 = np.zeros( tl1max+1, dtype=np.complex_ )
        
            for tl1 in range(tl1min, tl1max+1):
                #print(tl1)
                #print(qeXY.w[i][0][tl1])
                cl1[tl1] = qeXY.w[i][0][tl1] * cfunc_ZA( qeZA.w[j][i1_ZA][tl1] ) * (2.*tl1+1.) * fX[tl1]
                
    
            # l2 part
            tl2min = max(abs(qeXY.s[i][1]), abs(qeZA.s[j][i2_ZA]))
            tl2max = min( [qeXY.lmax, qeZA.lmax, lmax_fY] )

            cl2 = np.zeros( tl2max+1, dtype=np.complex_ )
    
            for tl2 in range(tl2min, tl2max+1):
                cl2[tl2] = qeXY.w[i][1][tl2] * cfunc_ZA( qeZA.w[j][i2_ZA][tl2] ) * (2.*tl2+1.) * fY[tl2]

            
            #glq  = scipy.special.roots_legendre()
            # transform l1 and l2 parts to position space
            glq = wignerd.gauss_legendre_quadrature( (tl1max + tl2max + lmax)/2 + 1 )
            gp1 = glq.cf_from_cl( qeXY.s[i][0], -(-1)**(conj_ZA)*qeZA.s[j][i1_ZA], cl1 )
            gp2 = glq.cf_from_cl( qeXY.s[i][1], -(-1)**(conj_ZA)*qeZA.s[j][i2_ZA], cl2 )

            # multiply and return to cl space
            clL = glq.cl_from_cf( lmax, qeXY.s[i][2], -(-1)**(conj_ZA)*qeZA.s[j][2], gp1 * gp2 )

            
            for L in range(0, lmax+1):
                ret[L] += clL[L] * qeXY.w[i][2][L] * cfunc_ZA( qeZA.w[j][2][L] ) / (32.*np.pi)
    
    #return ret
    
def fill_resp(qeXY, ret, fX, fY, qeZA=None):
    if qeZA==None: qeZA=qeXY
    return fill_resp_fullsky( qeXY, qeZA, ret, fX, fY)

