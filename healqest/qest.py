import logging
import sys
from typing import TypedDict

import numpy as np
import healpy as hp
from healqest import healqest_utils as utils
from healqest import weights, resp

logger = logging.getLogger(__name__)
np.seterr(all='ignore')


class qest(object):
    
    def __init__(self, config, cls, verbose=True):
        '''
        Set up the quadratic estimator calculation

        Parameters
        ----------
        config : dict
          Dictionary of lmin/lmax settings
        cls: dict
          Dictionary of cls
        '''
        
        # assert est=='lens' or est=='src' or est=='prof', "est expected to be lens/src/prof, got: %s"%est
        # assert cltype=='grad' or cltype=='len' or cltype=='unl', "cltype expected to be grad/len/unl, got: %s"%cltype
        if verbose:
            print('Setting up lensing reconstruction')
        self.config = config
        # self.almbar1 = almbar1
        # self.almbar2 = almbar2
        
        self.lmax = self.config['lensrec'][
            'Lmax']  # self.config['lensrec']['lmax'] = max(config['lensrec']['lmaxT'],config['lensrec']['lmaxP'])
        self.Lmax = self.config['lensrec']['Lmax']
        self.cltype = self.config['lensrec']['cltype']
        self.glm = {}
        self.clm = {}
        
        self.cls = cls
        
        if self.cltype!='ucmb' and self.cltype!='lcmb' and self.cltype!='gcmb':
            sys.exit('cltype must be ucmb, lcmb or gcmb')
        
        if 'nside' in self.config['lensrec']:
            if verbose:
                print("-- Overwrite default nside")
            self.nside = self.config['lensrec']['nside']  # Overwrite automatic setting of nside<2*lmax
            assert self.lmax<2.0*self.nside, "lmax must be less that 2*nside"
        else:
            self.nside = utils.get_nside(self.lmax)
        
        if verbose:
            print("-- Nside to project: %d"%self.nside)
            print("-- lmax:%d"%self.lmax)
            print("-- Lmax:%d"%self.Lmax)
            print("-- Using %s cls"%self.cltype)
    
    def eval(self, qe, almbar1, almbar2, u=None):
        '''
        Compute quadratic estimator

        Parameters
        ----------
        qe : str
          Quadratic estimator type: 'TT'/'EE'/'TE'/'EB'/'TB'/'TTprf'
        almbar1: complex array healpy alm
          First filtered alm
        almbar2: complex array healpy alm
          Second filtered alm
        u  : profile
          Profile instance

        Returns
        ----------
        glm: complex
          Gradient component of the plm
        clm:
          Curl component of the plm
        '''
        
        if qe in self.glm:
            print("We've already computed this!")
        else:
            # if qe is None:
            # sys.exit('Need to specify estimator')
            if qe=='TTprf' or qe=='TTmask' or qe=='TTnoise':
                assert u is not None, "Need profile function to compute this estimator"
            
            # def __init__(self,config,cls,est,u=None,totalcls=None):
            q = weights.weights(qe, self.cls[self.cltype], self.lmax, u=u)
            
            # sys.exit()
            print('Running lensing reconstruction')
            
            if qe[:2]=='TB' or qe[:2]=='EB':
                # Hack to get TB/EB working. currently not understanding some factors of j
                print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')
                
                wX1, wY1, wP1, sX1, sY1, sP1 = q.w[0][0], q.w[0][1], q.w[0][2], q.s[0][0], q.s[0][1], q.s[0][
                    2]
                wX3, wY3, wP3, sX3, sY3, sP3 = q.w[2][0], q.w[2][1], q.w[2][2], q.s[2][0], q.s[2][1], q.s[2][
                    2]
                
                walmbar1 = hp.almxfl(almbar1, wX1)  # T1/E1
                walmbar3 = hp.almxfl(almbar1, wX3)  # T3/E3
                walmbar2 = hp.almxfl(almbar2, wY1)  # B2
                
                SpX1, SmX1 = hp.alm2map_spin([walmbar1, np.zeros_like(walmbar1)], self.nside, 1, self.lmax)
                SpX3, SmX3 = hp.alm2map_spin([walmbar3, np.zeros_like(walmbar3)], self.nside, 3, self.lmax)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2), -1j*walmbar2], self.nside, 2,
                                             self.lmax)
                # SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)
                
                SpZ = SpY2*(SpX1 - SpX3) + SmY2*(SmX1 - SmX3)
                SmZ = -SpY2*(SmX1 + SmX3) + SmY2*(SpX1 + SpX3)
                
                glm, clm = hp.map2alm_spin([SpZ, SmZ], 1, self.Lmax)
                
                if qe=='TT' or qe=='EE' or qe=='TE' or qe=='ET':
                    nrm = 0.5
                elif qe=='EB':
                    nrm = -1
                else:
                    nrm = 1
                
                self.glm[qe] = hp.almxfl(glm, nrm*wP1)
                self.clm[qe] = hp.almxfl(clm, nrm*wP1)
            
            elif qe[:2]=='BT' or qe[:2]=='BE':
                # Hack to get TB/EB working. currently not understanding some factors of j
                print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')
                print('using est: %s'%qe)
                wX1, wY1, wP1, sX1, sY1, sP1 = q.w[0][0], q.w[0][1], q.w[0][2], q.s[0][0], q.s[0][1], q.s[0][
                    2]
                wX3, wY3, wP3, sX3, sY3, sP3 = q.w[2][0], q.w[2][1], q.w[2][2], q.s[2][0], q.s[2][1], q.s[2][
                    2]
                
                walmbar1 = hp.almxfl(almbar2, wY1)
                walmbar3 = hp.almxfl(almbar2, wY3)
                walmbar2 = hp.almxfl(almbar1, wX1)
                
                SpX1, SmX1 = hp.alm2map_spin([walmbar1, np.zeros_like(walmbar1)], self.nside, 1, self.lmax)
                SpX3, SmX3 = hp.alm2map_spin([walmbar3, np.zeros_like(walmbar3)], self.nside, 3, self.lmax)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2), -1j*walmbar2], self.nside, 2,
                                             self.lmax)
                # SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)
                
                SpZ = SpY2*(SpX1 - SpX3) + SmY2*(SmX1 - SmX3)
                SmZ = -SpY2*(SmX1 + SmX3) + SmY2*(SpX1 + SpX3)
                
                glm, clm = hp.map2alm_spin([SpZ, SmZ], 1, self.Lmax)
                
                if qe=='TT' or qe=='EE' or qe=='TE' or qe=='ET':
                    nrm = 0.5
                elif qe=='BE':
                    nrm = -1
                else:
                    nrm = 1
                
                self.glm[qe] = hp.almxfl(glm, nrm*wP1)
                self.clm[qe] = hp.almxfl(clm, nrm*wP1)
            
            elif qe=='TT2':
                q = weights.weights('TT', self.cls[self.cltype], self.lmax, u=u)
                
                # Hack to get TB/EB working. currently not understanding some factors of j
                print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')
                print('using est: %s'%qe)
                wX1, wY1, wP1, sX1, sY1, sP1 = q.w[0][0], q.w[0][1], q.w[0][2], q.s[0][0], q.s[0][1], q.s[0][
                    2]
                wX3, wY3, wP3, sX3, sY3, sP3 = q.w[2][0], q.w[2][1], q.w[2][2], q.s[2][0], q.s[2][1], q.s[2][
                    2]
                
                print('USING TOSHIYAS TT ESTIMATOR')
                # walmbar1 = hp.almxfl(almbar2,wY1)
                # walmbar3 = hp.almxfl(almbar2,wY3)
                # walmbar2 = hp.almxfl(almbar1,wX1)
                
                # SpX1, SmX1 = hp.alm2map_spin([walmbar1,np.zeros_like(walmbar1)], self.nside, 1, self.lmax)
                # SpX3, SmX3 = hp.alm2map_spin([walmbar3,np.zeros_like(walmbar3)], self.nside, 3, self.lmax)
                # SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)
                # SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)
                
                # SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                # SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)
                X = hp.alm2map(almbar1, self.nside)
                
                almbar2 = hp.almxfl(almbar2, q.w[0][0])
                Y, Z = hp.alm2map_spin([almbar2, 0*almbar2], self.nside, 1, self.lmax)
                XY = X*Y
                XZ = X*Z
                
                glm, clm = hp.map2alm_spin([XY, XZ], 1, self.Lmax)  # *q.w[0][2]
                # glm *=q.w[0][2]
                # clm *=q.w[0][2]
                
                # glm,clm = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)
                
                if qe=='TT' or qe=='EE' or qe=='TE' or qe=='ET':
                    nrm = 0.5
                elif qe=='BE':
                    nrm = -1
                else:
                    nrm = 1
                
                self.glm[qe] = hp.almxfl(glm, nrm*wP1)
                self.clm[qe] = hp.almxfl(clm, nrm*wP1)
                
                '''
                allocate(ilk(lmax)); ilk = 1d0
                if (gtype=='k') then
                    do l = 1, lmax
                    ilk(l) = 2d0/dble(l*(l+1))
                    end do
                end if

                ! compute convolution
                allocate(alm1(1,0:rlmax,0:rlmax))
                alm1 = 0d0
                do l = rlmin, rlmax
                    alm1(1,l,0:l) = Tlm1(l,0:l)
                end do 
                allocate(at(0:npix-1))
                call alm2map(nside,rlmax,rlmax,alm1,at)
                deallocate(alm1)

                allocate(alm1(2,0:rlmax,0:rlmax))
                alm1 = 0d0
                do l = rlmin, rlmax
                    alm1(1,l,0:l) = fC(l)*Tlm2(l,0:l)*dsqrt(dble((l+1)*l))
                end do 
                allocate(map(0:npix-1,2))
                call alm2map_spin(nside,rlmax,rlmax,1,alm1,map)
                map(:,1) = at*map(:,1)
                map(:,2) = at*map(:,2)
                deallocate(at,alm1)

                allocate(blm(2,0:lmax,0:lmax))
                call map2alm_spin(nside,lmax,lmax,1,map,blm)
                deallocate(map)

                ! compute glm and clm
                glm = 0d0
                clm = 0d0
                do l = 1, lmax
                    glm(l,0:l) = ilk(l)*dsqrt(dble(l*(l+1)))*blm(1,l,0:l)
                    clm(l,0:l) = ilk(l)*dsqrt(dble(l*(l+1)))*blm(2,l,0:l)
                end do
                ''';
            
            else:
                # More traditional quicklens style calculation
                retglm = 0
                retclm = 0
                
                for i in range(0, q.ntrm):
                    
                    wX, wY, wP, sX, sY, sP = q.w[i][0], q.w[i][1], q.w[i][2], q.s[i][0], q.s[i][1], q.s[i][2]
                    # print("-- Computing term %d/%d, sj = [%d,%d,%d]"%(i+1,q.ntrm,sX,sY,sP))
                    walmbar1 = hp.almxfl(almbar1, wX)
                    walmbar2 = hp.almxfl(almbar2, wY)
                    
                    # Input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
                    # -----------------------------------------------------------------------------------------------
                    
                    if qe[0]=='B':
                        SpX, SmX = hp.alm2map_spin([np.zeros_like(walmbar1), 1j*walmbar1], self.nside,
                                                   np.abs(sX), self.lmax)
                        sys.exit('broken')
                    else:
                        SpX, SmX = hp.alm2map_spin([walmbar1, np.zeros_like(walmbar1)], self.nside,
                                                   np.abs(sX), self.lmax)
                    
                    X = SpX + 1j*SmX  # Complex map _{+s}S or _{-s}S
                    
                    if sX<0:
                        X = np.conj(X)*(-1)**(sX)
                    # -----------------------------------------------------------------------------------------------
                    if qe[1]=='B':
                        SpY, SmY = hp.alm2map_spin([np.zeros_like(walmbar2), 1j*walmbar2], self.nside,
                                                   np.abs(sY), self.lmax)
                        sys.exit('broken')
                    else:
                        SpY, SmY = hp.alm2map_spin([walmbar2, np.zeros_like(walmbar2)], self.nside,
                                                   np.abs(sY), self.lmax)
                    
                    Y = SpY + 1j*SmY
                    
                    if sY<0:
                        Y = np.conj(Y)*(-1)**(sY)
                    
                    # -----------------------------------------------------------------------------------------------
                    
                    XY = X*Y
                    
                    if sP<0:
                        XY = np.conj(XY)*(-1)**(sP)
                    
                    glm, clm = hp.map2alm_spin([XY.real, XY.imag], np.abs(sP), self.Lmax)
                    
                    glm = hp.almxfl(glm, 0.5*wP)
                    clm = hp.almxfl(clm, 0.5*wP)
                    
                    retglm += glm
                    retclm += clm
                
                self.glm[qe] = retglm
                self.clm[qe] = retclm
                # if qe == self.qe:
                #    self.retglm = retglm
                #    self.retclm = retclm
                # elif qe == 'TTprf':
                #    self.retglm_prf = retglm
                #    self.retclm_prf = retclm
        
        return self.glm[qe], self.clm[qe]
    
    def get_aresp(self, flX, flY, qe1=None, qe2=None, u=None):
        '''
        Compute analytical response function for 1D filtering

        Parameters
        ----------
        flX, flY
          1D real arrays representing the filter functions for the X and Y fields
        qe1: string
          First estimator
        qe2: string
          Second estimator; if None, assumes it is the same as qe1

        Returns
        ----------
        aresp:
          Analytical response function
        '''
        if qe1 is None:
            assert 0, "qe1 must be defined"
        
        qeXY = weights.weights(qe1, self.cls[self.cltype], self.lmax, u=u)
        
        if qe2 is None or qe2==qe1:
            qeZA = None
        else:
            qeZA = weights.weights(qe2, self.cls[self.cltype], self.lmax, u=u)
        
        aresp = resp.fill_resp(qeXY, np.zeros(self.Lmax + 1, dtype=np.complex_), flX, flY, qeZA=qeZA)
        
        return aresp
    
    def harden(self, qe, almbar1, almbar2, flX, flY, u, qe_hrd='TTprf'):
        '''
        Get the source hardened glm and the response function.
        Need arguments flX, flY in order to compute the analytical response
        needed for hardening.

        Parameters
        ----------
        flX, flY
          1D real arrays representing the filter functions for the X and Y fields

        Returns
        ----------
        plm :
          Source hardened glm
        resp :
          Response function
        '''
        assert qe=='TT', "We only harden for qe 'TT', got: %s"%qe
        
        ss = self.get_aresp(flX, flY, qe1=qe_hrd, u=u)
        es = self.get_aresp(flX, flY, qe1=qe_hrd, qe2=qe, u=u)
        ee = self.get_aresp(flX, flY, qe1=qe)
        
        plm1, _ = self.eval(qe, almbar1, almbar2)
        plm2, _ = self.eval(qe_hrd, almbar1, almbar2, u)
        
        weight = -1*es/ss
        plm = plm1 + hp.almxfl(plm2, weight)
        resp = ee + weight*es
        
        return plm, resp


class CMBCl(TypedDict):
    tt: np.ndarray
    te: np.ndarray
    ee: np.ndarray
    bb: np.ndarray


class Qest(qest):
    """
    QE estimator following the cmblensplus convention.
    """
    __prf_estimators__ = ['TTprf', 'EEprf', 'TEprf', 'ETprf']  # exclude the odd parity ones!
    
    def __init__(self, lmax, Lmax, Cls, nside=None, flT=None, flP=None, gmv=False):
        """
        Parameters
        ----------
        lmax: int
            Maximum multipole of the cmb map alm
        Lmax: int
            Maximum multipole of the lens rec alm
        Cls: CMBCl
            dict of cls
        nside: int=None
            Healpix nside used for intermediate alm2map operations,
            the nside has to be large enough, >1/2 lmax, to avoid aliasing
        flT, flP: np.ndarray=None
            binary array of shape (lmax+1, ), indicating the POST cinv ell selection.
        gmv: bool=False
            If True, the response function and prf-hardening follows the GMV formalism
        """
        self.lmax = lmax
        self.Lmax = Lmax
        self.size = hp.Alm.getsize(self.lmax)
        # Post cinv ell cut
        self.fl_cut = dict()
        self.fl_cut['T'] = flT[:self.lmax + 1] if flT is not None else np.ones(self.lmax + 1)
        self.fl_cut['E'] = flP[:self.lmax + 1] if flP is not None else np.ones(self.lmax + 1)
        self.fl_cut['B'] = flP[:self.lmax + 1] if flP is not None else np.ones(self.lmax + 1)
        assert self.fl_cut['T'].shape[-1]==self.lmax + 1
        assert self.fl_cut['E'].shape[-1]==self.lmax + 1
        assert self.fl_cut['B'].shape[-1]==self.lmax + 1
        
        self.cls = Cls
        if nside is None:
            nside = utils.get_nside(lmax)
        self.nside = nside
        
        assert self.lmax<2.0*self.nside, "lmax must be less that 2*nside"
        self.gmv = gmv
    
    @staticmethod
    def alm2map_spin(alm, fell, nside, spin, lmax, mmax=None, g=None):
        """ convert a spin-0 alm into a complex spin field (Q +/- iU): out = Q, +/-U"""
        if spin==0:
            walm = hp.almxfl(alm, fell)
            # alm2map is recommended over alm2map_spin for spin=0
            if g is None:
                out = hp.alm2map(walm, nside=nside, lmax=lmax, mmax=mmax)
            else:
                out = g.alm2map(walm, lmax=lmax, mmax=mmax)
            return out, 0
        else:
            zero = np.zeros_like(alm)
            _fell = (-1)**spin*np.conj(fell) if spin<0 else fell
            _fell *= -1
            if np.all(fell.imag==0):
                E = hp.almxfl(alm, _fell.real)
                B = zero
            elif np.all(fell.real==0):
                E = zero
                B = hp.almxfl(alm, _fell.imag)
            else:
                raise ValueError("Fell must be real or imaginary")
            if g is None:
                q, u = hp.alm2map_spin([E, B], nside=nside, spin=np.abs(spin), lmax=lmax, mmax=mmax)
            else:
                q, u = g.alm2map_spin([E, B], spin=np.abs(spin), lmax=lmax, mmax=mmax)
            if spin>0:
                return q, u
            else:
                return q, -u
    
    def eval(self, qe, almbar1, almbar2, u=None, g=None, distortion='lens'):
        """
        Compute quadratic estimator

        Parameters
        ----------
        qe: str
          Quadratic estimator type (defined in `weights_plus`): 'TT'/'EE'/'TE'/'EB'/'TB'/'prf'
        almbar1,almbar2: complex array healpy alm
          First and second filtered alm
        u: np.ndarray=None
          Profile instance
        g: Geometry
            Geometry instance defined within declination range. This will be used
            to compute spherical harmonics functions with ducc0. If None, the slower
            full-sky healpy functions will be used.
        distortion: str
            distortion type, 'lens' or 'prf' or 'tau'
        Returns
        -------
        glm, clm: tuple of complex array
            Gradient/curl component of the plm
        """
        assert almbar1.shape[-1]==self.size, f"almbar size {almbar1.shape[-1]} don't match lmax {self.lmax})"
        assert almbar2.shape[-1]==self.size, f"almbar size {almbar2.shape[-1]} don't match lmax {self.lmax})"
        
        if distortion in ['prf']:
            assert u is not None, "Need profile function to compute this estimator"
        
        q = weights.weights_plus(qe, self.cls, self.lmax, u=u, distortion=distortion)
        
        logger.info(f'Running {distortion} reconstruction: {qe}')
        
        retglm = 0
        retclm = 0
        if g is not None:
            assert g.nside==self.nside
        assert q.ntrm%2==0, f"Number of terms must be even: {q.ntrm}"
        for i in range(0, q.ntrm//2):
            # skipping second half of reducant terms
            wX, wY, wP, sX, sY, sP = q.w[i][0], q.w[i][1], q.w[i][2], q.s[i][0], q.s[i][1], q.s[i][2]
            
            Xq, Xu = self.alm2map_spin(almbar1, fell=wX, nside=self.nside, spin=sX, lmax=self.lmax, g=g)
            Yq, Yu = self.alm2map_spin(almbar2, fell=wY, nside=self.nside, spin=sY, lmax=self.lmax, g=g)
            XYq = Xq*Yq - Xu*Yu  # XY = X*Y
            XYu = Xq*Yu + Yq*Xu  # XY = X*Y
            
            if np.all(wP.imag==0):
                _wP = wP
            elif np.all(wP.real==0):
                # swap grad/curl mode such that glm is curl and clm is grad
                # wP has an -1j factor, here we move the factor from wP to XY.
                _wP = wP*1j
                XYq, XYu = XYu, -XYq  # XY *=-1j
            else:
                raise ValueError("wP must be real or imaginary")
            if sP<0:
                # This is for the second half reduncant transform, we normally don't end up here.
                # XY = np.conj(XY) * (-1) ** sP  # because wP has a (-1)**sP factor, here we are canceling it.
                XYq *= (-1)**sP  # XY = np.conj(XY) * (-1) ** sP
                XYu *= -(-1)**sP  # XY = np.conj(XY) * (-1) ** sP
            
            if g is None:
                glm, clm = hp.map2alm_spin([XYq, XYu], np.abs(sP), self.Lmax)
            else:
                glm, clm = g.map2alm_spin([XYq, XYu], spin=np.abs(sP), lmax=self.Lmax, check=False, )
            glm = hp.almxfl(glm, _wP)
            clm = hp.almxfl(clm, _wP)  # for curl est, this will be -grad.
            
            retglm += glm
            retclm += clm
        
        return retglm, retclm
    
    # def get_aresp(self, flX, flY, qe1=None, qe2=None, u=None, fast=False, curl=False):
    #     """
    #     Compute analytical response function for 1D filtering
    #
    #     Parameters
    #     ----------
    #     flX, flY
    #       1D real arrays representing the filter functions for the X and Y fields
    #     qe1: string
    #       First estimator
    #     qe2: string
    #       Second estimator; if None, assumes it is the same as qe1
    #     fast: bool=False
    #         If True, uses the fast response function calculation.
    #     curl: bool=False
    #         If True, `qe1` and `qe2` are suffixed with `curl` to compute curl-mode response.
    #     Returns
    #     ----------
    #     aresp:
    #       Analytical response function
    #     """
    #     if qe1 is None:
    #         assert 0, "qe1 must be defined"
    #
    #     qeXY = weights.weights_plus(qe1 if not curl else f'{qe1}curl', self.cls, self.lmax, u=u)
    #
    #     if qe2 is None or qe2 == qe1:
    #         qeZA = qeXY
    #     else:
    #         qeZA = weights.weights_plus(qe2 if not curl else f'{qe2}curl', self.cls, self.lmax, u=u)
    #
    #     aresp = resp.fill_resp_fullsky(qeXY, qeZA, np.zeros(self.Lmax + 1, dtype=complex), flX, flY, fast=fast)
    #     return aresp
    
    # def harden(self, qe, almbar1, almbar2, flX, flY, u, qe_hrd='prf', curl=False):
    #     """
    #     Get the source hardened glm and the response function.
    #     Need arguments flX, flY in order to compute the analytical response
    #     needed for hardening.
    #
    #     Parameters
    #     ----------
    #     qe: string
    #       Quadratic estimator type, only 'TT' is supported
    #     flX, flY
    #       1D real arrays representing the filter functions for the X and Y fields
    #
    #     Returns
    #     ----------
    #     plm :
    #       Source hardened glm
    #     resp :
    #       Response function
    #     """
    #     assert qe == 'TT', f"We only harden for qe 'TT', got: {qe}"
    #
    #     ss = self.get_aresp(flX, flY, qe1=qe_hrd, u=u)
    #     es = self.get_aresp(flX, flY, qe1=qe_hrd, qe2=qe, u=u, fast=False)
    #     ee = self.get_aresp(flX, flY, qe1=qe, curl=curl)
    #
    #     plm1 = self.eval(qe, almbar1, almbar2)[1 if curl else 0]
    #     plm2 = self.eval(qe_hrd, almbar1, almbar2, u)[0]
    #
    #     weight = -1 * es / ss
    #     plm = plm1 + hp.almxfl(plm2, weight)
    #     R = ee + weight * es
    #     return plm, R
    
    def fls2fls_dict(self, fls):
        from healqest.cinv.cinv_utils import cli
        clte = cli(fls[3, :self.lmax + 1]) if self.gmv else np.zeros(self.lmax + 1)
        if self.gmv:
            assert np.any(clte>0)
        inv = fls[0, :self.lmax + 1]*fls[1, :self.lmax + 1]
        inv = inv*cli(1 - inv*clte**2)
        
        fl = dict()
        fl['BB'] = fls[2, :self.lmax + 1]
        fl['TT'] = inv*cli(fls[1, :self.lmax + 1])
        fl['EE'] = inv*cli(fls[0, :self.lmax + 1])
        fl['TE'] = fl['ET'] = -inv*clte
        
        return fl
    
    # def get_aresp_gmv(self, fls, qe=None, u=None, fast=False, curl=False, TTprf_type=None):
    #     """
    #     Compute analytical response function for GMV (jointly filtered maps)
    #
    #     Parameters
    #     ----------
    #     qe: str
    #         Quadratic estimator type.
    #     fls: np.array
    #         shape: (4, lmax+1), filter functions for TT/EE/BB/TE, i.e., 1/Cltt, 1/Clee, 1/Clbb, 1/Clte.
    #     u: np.ndarray
    #         shape (lmax+1, ) profile function for TTprf estimator
    #     fast: bool=False
    #         If True, uses the fast response function calculation.
    #     curl: bool=False
    #         If True, `qe` is suffixed with `curl` to compute curl-mode response.
    #     TTprf_type: str=None
    #         type of the prf estimator, one of ['ss', 'es', 'se'].
    #     """
    #     if TTprf_type is not None:
    #         assert TTprf_type in ['ss', 'es', 'se']
    #         assert u is not None, "Need profile function to compute this estimator"
    #
    #     fl = self.fls2fls_dict(fls)
    #
    #     s1, s2 = qe[0], qe[1]
    #     assert s1 in 'TEB' and s2 in 'TEB', f"qe must be one of TEB, got: {qe}"
    #
    #     if self.gmv:
    #         keys = list(fl.keys())
    #     else:
    #         keys = [f"{s1}{s1}", f"{s2}{s2}"]  # SQE only picks the 2 (can be the same) diagonal terms.
    #
    #     if TTprf_type in ['ss', 'es']:
    #         loop1 = loop2 = 'T'
    #     else:
    #         loop1 = loop2 = 'TEB'
    #     if TTprf_type in ['ss', 'se']:
    #         qeXY = weights.weights_plus('prf', self.cls, self.lmax, u=u)
    #     else:
    #         qeXY = weights.weights_plus(qe if not curl else f"{qe}curl", self.cls, self.lmax)
    #
    #     R = np.zeros(self.Lmax+1, dtype=float)
    #     for _s1 in loop1:
    #         _qe1 = s1+_s1
    #         if _qe1 not in keys:
    #             continue
    #         flX = fl[_qe1]*self.fl_cut[s1]
    #         for _s2 in loop2:
    #             _qe2 = s2 + _s2
    #
    #             if _qe2 not in keys:
    #                 continue
    #             flY = fl[_qe2]*self.fl_cut[s2]
    #             if TTprf_type in ['ss', 'es']:
    #                 qeZA = weights.weights_plus('prf', self.cls, self.lmax, u=u)
    #             else:
    #                 qeZA = weights.weights_plus(_s1+_s2 if not curl else f"{_s1+_s2}curl", self.cls, self.lmax)
    #             _R = resp.fill_resp_fullsky(qeXY, qeZA, np.zeros(self.Lmax+1, dtype=complex), flX, flY, fast=fast)
    #             R += _R
    #     return R
    
    def get_resp(self, fls, qe, fls2=None, u=None, fast=False, curl=False, type1='lens', type2=None):
        """
        Compute the cross response between two estimators. Assume joint cinv filtering.

        Note
        ----
        For example, we want to see how much the f^XY estimator extract the distrotion field (encoded by weights g)
        from filtered maps \bar{X} and \bar{Y}, which can be decomposed into W^{XZ} Z and W^{YA} A,
        where X/Y/Z/A are T/E/B, and W are the filter functions (gmv, or sqe in the diagonal case).
        In the general form, this is computing f^{XY} W^{XZ} W^{YA} g^{ZA}. Note that, some terms in g might not
        exist, e.g. non-TT terms for the profile estimator.

        Parameters
        ----------
        fls, fls2: array of shape (4, lmax+1)
            filter functions for TT/EE/BB/TE, i.e., 1/Cl
        qe: str
            Quadratic estimator type, e.g., 'TT','EB'
        u: np.ndarray=None
            profile function for prf estimator
        fast: bool
            If True, uses the fast response function calculation.
        curl: bool
            If True, `qe` is suffixed with `curl` to compute curl-mode response.
        type1, type2: str
            distortion field  type for the estimator, 'lens' or 'prf' or 'tau' or 'rot'.

        Returns
        -------
        R: np.ndarray
            response function
        """
        
        if type2 is None:
            type2 = type1
        fl1 = self.fls2fls_dict(fls)
        if fls2 is None:
            fl2 = fl1
        else:
            fl2 = self.fls2fls_dict(fls2)
        s1, s2 = qe[0], qe[1]
        assert s1 in 'TEB' and s2 in 'TEB', f"qe must be one of TEB, got: {qe}"
        
        if self.gmv:
            keys = list(fl1.keys())
        else:
            keys = [f"{s1}{s1}", f"{s2}{s2}"]  # SQE only picks the 2 (can be the same) diagonal terms.
        
        R = np.zeros(self.Lmax + 1, dtype=float)
        if qe not in weights.weights_plus.estimators(type1):
            logger.warning(f"{type1} distortion does not have {qe} defined. set response to 0.")
            return R
        else:
            qeXY = weights.weights_plus(qe, self.cls, self.lmax, distortion=type1, curl=curl, u=u)
        
        for _s1 in 'TEB':
            _qe1 = s1 + _s1
            if _qe1 not in keys:
                continue
            flX = fl1[_qe1]*self.fl_cut[s1]
            for _s2 in 'TEB':
                _qe2 = s2 + _s2
                if _qe2 not in keys:
                    continue
                flY = fl2[_qe2]*self.fl_cut[s2]
                
                if _s1 + _s2 not in weights.weights_plus.estimators(type2):
                    # sometimes `_qe2` is not defined for the second distortion field,
                    # in this case it should be skipped.
                    logger.warning(f"{type2} distortion field does not have {_s1}{_s2} defined.")
                    continue
                else:
                    qeZA = weights.weights_plus(_s1 + _s2, self.cls, self.lmax, distortion=type2, curl=curl,
                                                u=u)
                    _R = resp.fill_resp_fullsky(qeXY, qeZA, np.zeros(self.Lmax + 1, dtype=complex), flX, flY,
                                                fast=fast)
                    R += _R
        return R
    
    def get_harden_weights(self, qe, fls, u, fls2=None, curl=False, fast=False, type1='lens', type2='prf'):
        assert type2=='prf', "This is implemented for TTprf only (for any estimator)."
        # ss = self.get_aresp_gmv(fls, qe="TT", u=u, fast=fast, curl=False, TTprf_type='ss')
        # es = self.get_aresp_gmv(fls, qe=qe, u=u, fast=fast, curl=curl, TTprf_type='es')
        # se = self.get_aresp_gmv(fls, qe="TT", u=u, fast=fast, curl=curl, TTprf_type='se')
        es = self.get_resp(fls, qe, curl=curl, fast=fast, type1=type1, type2=type2, u=u, fls2=fls2)
        ss = self.get_resp(fls, 'TT', curl=False, fast=fast, type1=type2, type2=type2, u=u, fls2=fls2)
        se = self.get_resp(fls, 'TT', curl=curl, fast=fast, type1=type2, type2=type1, u=u, fls2=fls2)
        weight = -1*es/ss
        return weight, se
    
    def rec_and_resp(self, qe, almbars1, almbars2, fls, fls2=None, u=None, g=None, fast=False, type1='lens'):
        """
        compute lensing reconstruction for grad and curl modes, return also the analytical response functions.

        Parameters
        ----------
        qe: str
            Quadratic estimator type, e.g., 'TT','TTprf'
        almbars1, almbars2: complex arrays
            First and second filtered alms, shape (3, nalm)
        fls, fls2: np.ndarray.
            shape: (4, lmax+1), filter functions for TT/EE/BB/TE. If the two are the same, then set fls2=None.
        u: np.ndarray=None
            profile function for TTprf estimator
        g: Geometry=None
            Geometry instance defined within declination range.
        fast: bool=False
            If True, uses the fast response function calculation.
        type1: str
            distortion field  type for the estimator, 'lens' or 'prf' or 'tau'

        Returns
        -------
        [glm, clm]: list of complex array
            Gradient/curl component of the plm
        [aresp_g, aresp_c]: list of np.ndarray
            Analytical response function for grad/curl mode
        hrd_out: dict or None
            If `qe` ends with 'prf', return a dict containing the source response functions
        """
        
        i1 = 'teb'.index(qe[0].lower())
        i2 = 'teb'.index(qe[1].lower())
        
        if qe.endswith('prf'):
            _qe = qe.removesuffix('prf')
        else:
            _qe = qe
        
        glm, clm = self.eval(_qe, almbars1[i1], almbars2[i2], g=g, distortion=type1)
        # aresp_g = self.get_aresp_gmv(fls, _qe, fast=fast)
        # aresp_c = self.get_aresp_gmv(fls, _qe, fast=fast, curl=True)
        aresp_g = self.get_resp(fls, _qe, fast=fast, type1=type1, type2=type1, fls2=fls2)
        if type1=='lens':
            aresp_c = self.get_resp(fls, _qe, fast=fast, curl=True, type1=type1, type2=type1, fls2=fls2)
        else:
            aresp_c = np.zeros_like(aresp_g)
        
        # do the source harden stuff
        if qe.endswith('prf'):
            if not self.gmv:
                assert _qe=='TT', f"We only harden for 'TT' for SQE, got: {qe}"
            slm = self.eval('TT', almbars1[0], almbars2[0], u=u, g=g, distortion='prf')[0]
            w_g, se_g = self.get_harden_weights(_qe, fls, u, curl=False, fast=fast, type1=type1, type2='prf',
                                                fls2=fls2)
            if type1=='lens':
                w_c, se_c = self.get_harden_weights(_qe, fls, u, curl=True, fast=fast, type1=type1,
                                                    type2='prf',
                                                    fls2=fls2)
            else:
                w_c = np.zeros_like(w_g)
                se_c = np.zeros_like(se_g)
            
            glm += hp.almxfl(slm, w_g)
            clm += hp.almxfl(slm, w_c)
            aresp_g += w_g*se_g
            aresp_c += w_c*se_c
        
        return [glm, clm], [aresp_g, aresp_c]


# Generalized bias-hardening tools
def det(idx_i, idx_j, func):
    """
    compute determinant of submat, with row, col index specified by idx_i and idx_j
    """
    if len(idx_i)==len(idx_j)==1:
        return func(idx_i[0], idx_j[0])
    tot = 0
    for k, j in enumerate(idx_j):
        tot += func(idx_i[0], j)*cofactor(idx_i, idx_j, 0, k, func=func)
    return tot


def cofactor(idx_i, idx_j, ki, kj, func):
    """
    compute cofactor of submat, with row, col index specified by idx_i and idx_j, for element (ki, kj)
    """
    # sometimes we run into computing cofactor of a 1x1 matrix.
    if len(idx_i)==len(idx_j)==1:
        return np.ones_like(func(idx_i[0], idx_j[0]))
    sign = (-1)**(ki + kj)
    i_sub = list(idx_i.copy())
    j_sub = list(idx_j.copy())
    i_sub.pop(ki)
    j_sub.pop(kj)
    det_minor = det(i_sub, j_sub, func)
    return sign*det_minor


def get_harden_resps(keys, func):
    """
    Example
    -------
    >>> func = lambda k1, k2: np.sum([R[f'{k1}-{k2}'][qe] for qe in ['TT', 'EE', 'TE', 'ET']], axis=0)
    >>> get_harden_resps([ 'tau', 'lens', 'prf'], func=func)
    """
    
    R = 0
    C11 = cofactor(keys, keys, 0, 0, func)
    for j, k in enumerate(keys):
        Rk = func(keys[0], k)
        Ck = cofactor(keys, keys, 0, j, func)
        R += Ck*Rk/C11
    return R


def get_harden_weights(keys, i, func):
    """
    Example
    -------
    >>> keys = ['tau', 'lens', 'prf']
    >>> func = lambda k1, k2: np.sum([R[f'{k1}-{k2}'][qe] for qe in ['TT', 'EE', 'TE', 'ET']], axis=0)
    >>> w_len = get_harden_weights(keys, 1, func=func) # weights for the lens component to harden against tau
    """
    C11 = cofactor(keys, keys, 0, 0, func)
    Ck = cofactor(keys, keys, i, 0, func)
    return Ck/C11
