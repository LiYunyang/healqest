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

        #assert est=='lens' or est=='src' or est=='prof', "est expected to be lens/src/prof, got: %s"%est
        #assert cltype=='grad' or cltype=='len' or cltype=='unl', "cltype expected to be grad/len/unl, got: %s"%cltype
        if verbose:
            print('Setting up lensing reconstruction')
        self.config  = config
        #self.almbar1 = almbar1
        #self.almbar2 = almbar2

        self.lmax    = self.config['lensrec']['Lmax']#self.config['lensrec']['lmax'] = max(config['lensrec']['lmaxT'],config['lensrec']['lmaxP'])
        self.Lmax    = self.config['lensrec']['Lmax']
        self.cltype  = self.config['lensrec']['cltype']
        self.glm = {}
        self.clm = {}

        self.cls     = cls

        if self.cltype!='ucmb' and self.cltype!='lcmb' and self.cltype!='gcmb':
            sys.exit('cltype must be ucmb, lcmb or gcmb')

        if 'nside' in self.config['lensrec']:
            if verbose:
                print("-- Overwrite default nside")
            self.nside = self.config['lensrec']['nside'] # Overwrite automatic setting of nside<2*lmax
            assert self.lmax < 2.0*self.nside, "lmax must be less that 2*nside"
        else:
            self.nside = utils.get_nside(self.lmax)

        if verbose:
            print("-- Nside to project: %d"%self.nside)
            print("-- lmax:%d"%self.lmax)
            print("-- Lmax:%d"%self.Lmax)
            print("-- Using %s cls"%self.cltype)

    def eval(self,qe,almbar1,almbar2,u=None):
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
            #if qe is None:
            #sys.exit('Need to specify estimator')
            if qe == 'TTprf' or qe == 'TTmask' or qe == 'TTnoise':
                assert u is not None, "Need profile function to compute this estimator"

            #def __init__(self,config,cls,est,u=None,totalcls=None):
            q = weights.weights(qe, self.cls[self.cltype], self.lmax, u=u)

            #sys.exit()
            print('Running lensing reconstruction')

            if qe[:2]=='TB' or qe[:2]=='EB':
                # Hack to get TB/EB working. currently not understanding some factors of j
                print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')

                wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
                wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

                walmbar1 = hp.almxfl(almbar1,wX1) # T1/E1
                walmbar3 = hp.almxfl(almbar1,wX3) # T3/E3
                walmbar2 = hp.almxfl(almbar2,wY1) # B2

                SpX1, SmX1 = hp.alm2map_spin([walmbar1,np.zeros_like(walmbar1)], self.nside, 1, self.lmax)
                SpX3, SmX3 = hp.alm2map_spin([walmbar3,np.zeros_like(walmbar3)], self.nside, 3, self.lmax)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)
                #SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)

                SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                glm,clm = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                if qe=='TT' or qe=='EE' or qe=='TE' or qe=='ET':
                    nrm=0.5
                elif qe=='EB':
                    nrm=-1
                else:
                    nrm=1

                self.glm[qe] = hp.almxfl(glm,nrm*wP1)
                self.clm[qe] = hp.almxfl(clm,nrm*wP1)

            elif qe[:2]=='BT' or qe[:2]=='BE':
                # Hack to get TB/EB working. currently not understanding some factors of j
                print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')
                print('using est: %s'%qe )
                wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
                wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

                walmbar1 = hp.almxfl(almbar2,wY1)
                walmbar3 = hp.almxfl(almbar2,wY3)
                walmbar2 = hp.almxfl(almbar1,wX1)

                SpX1, SmX1 = hp.alm2map_spin([walmbar1,np.zeros_like(walmbar1)], self.nside, 1, self.lmax)
                SpX3, SmX3 = hp.alm2map_spin([walmbar3,np.zeros_like(walmbar3)], self.nside, 3, self.lmax)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)
                #SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)

                SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                glm,clm = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                if qe=='TT' or qe=='EE' or qe=='TE' or qe=='ET':
                    nrm=0.5
                elif qe=='BE':
                    nrm=-1
                else:
                    nrm=1

                self.glm[qe] = hp.almxfl(glm,nrm*wP1)
                self.clm[qe] = hp.almxfl(clm,nrm*wP1)

            elif qe=='TT2':
                q = weights.weights('TT', self.cls[self.cltype], self.lmax, u=u)

                # Hack to get TB/EB working. currently not understanding some factors of j
                print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')
                print('using est: %s'%qe )
                wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
                wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

                print('USING TOSHIYAS TT ESTIMATOR')
                #walmbar1 = hp.almxfl(almbar2,wY1)
                #walmbar3 = hp.almxfl(almbar2,wY3)
                #walmbar2 = hp.almxfl(almbar1,wX1)

                #SpX1, SmX1 = hp.alm2map_spin([walmbar1,np.zeros_like(walmbar1)], self.nside, 1, self.lmax)
                #SpX3, SmX3 = hp.alm2map_spin([walmbar3,np.zeros_like(walmbar3)], self.nside, 3, self.lmax)
                #SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)
                #SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)

                #SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                #SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)
                X = hp.alm2map(almbar1,self.nside)

                almbar2 = hp.almxfl(almbar2,q.w[0][0])
                Y,Z = hp.alm2map_spin([almbar2,0*almbar2],self.nside,1,self.lmax  )
                XY  = X*Y
                XZ  = X*Z

                glm,clm = hp.map2alm_spin([XY,XZ],1,self.Lmax)#*q.w[0][2]
                #glm *=q.w[0][2]
                #clm *=q.w[0][2]
                
                #glm,clm = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                if qe=='TT' or qe=='EE' or qe=='TE' or qe=='ET':
                    nrm=0.5
                elif qe=='BE':
                    nrm=-1
                else:
                    nrm=1

                self.glm[qe] = hp.almxfl(glm,nrm*wP1)
                self.clm[qe] = hp.almxfl(clm,nrm*wP1)

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
                retglm  = 0
                retclm  = 0

                for i in range(0,q.ntrm):

                    wX,wY,wP,sX,sY,sP = q.w[i][0],q.w[i][1],q.w[i][2],q.s[i][0],q.s[i][1],q.s[i][2]
                    #print("-- Computing term %d/%d, sj = [%d,%d,%d]"%(i+1,q.ntrm,sX,sY,sP))
                    walmbar1 = hp.almxfl(almbar1,wX)
                    walmbar2 = hp.almxfl(almbar2,wY)

                    # Input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
                    #-----------------------------------------------------------------------------------------------

                    if qe[0]=='B':
                        SpX, SmX = hp.alm2map_spin([np.zeros_like(walmbar1),1j*walmbar1],self.nside,np.abs(sX),self.lmax)
                        sys.exit('broken')
                    else:
                        SpX, SmX = hp.alm2map_spin([walmbar1,np.zeros_like(walmbar1)],self.nside,np.abs(sX),self.lmax)

                    X  = SpX+1j*SmX # Complex map _{+s}S or _{-s}S

                    if sX<0:
                        X = np.conj(X)*(-1)**(sX)
                    #-----------------------------------------------------------------------------------------------
                    if qe[1]=='B':
                        SpY, SmY = hp.alm2map_spin([np.zeros_like(walmbar2),1j*walmbar2],self.nside,np.abs(sY),self.lmax)
                        sys.exit('broken')
                    else:
                        SpY, SmY = hp.alm2map_spin([walmbar2,np.zeros_like(walmbar2)],self.nside,np.abs(sY),self.lmax)

                    Y  = SpY+1j*SmY

                    if sY<0:
                        Y = np.conj(Y)*(-1)**(sY)

                    #-----------------------------------------------------------------------------------------------

                    XY = X*Y

                    if sP<0:
                        XY = np.conj(XY)*(-1)**(sP)

                    glm,clm  = hp.map2alm_spin([XY.real,XY.imag], np.abs(sP), self.Lmax)

                    glm = hp.almxfl(glm,0.5*wP)
                    clm = hp.almxfl(clm,0.5*wP)

                    retglm  += glm
                    retclm  += clm

                self.glm[qe] = retglm
                self.clm[qe] = retclm
                #if qe == self.qe:
                #    self.retglm = retglm
                #    self.retclm = retclm
                #elif qe == 'TTprf':
                #    self.retglm_prf = retglm
                #    self.retclm_prf = retclm

        return self.glm[qe], self.clm[qe]

    def get_aresp(self,flX,flY,qe1=None,qe2=None,u=None):
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

        plm1,_ = self.eval(qe,almbar1,almbar2)
        plm2,_ = self.eval(qe_hrd,almbar1,almbar2,u)

        weight = -1*es/ss
        plm    = plm1 + hp.almxfl(plm2, weight)
        resp   = ee + weight*es

        return plm, resp

class qest_gmv(object):

    def __init__(self,config,cls):
        '''
        Set up the quadratic estimator calculation for GMV

        Parameters
        ----------
        config : dict
          Dictionary of settings
        qe  : str
          Quadratic estimator type: 'all'/'TTEETE' (TT/EE/TE only)/'TBEB' (TB/EB only)
        alm1all: complex
          First unfiltered alms; N x 5 arrays for each of the 5 estimators in the order TT/EE/TE/TB/EB
        alm2all: complex
          Second unfiltered alms; N x 5 arrays for each of the 5 estimators in the order TT/EE/TE/TB/EB
        totalcls:
          The signal + noise spectra for TT, EE, BB, TE needed for the weights
        cltype : str
          Should be one of 'grad'/'len'/'unl'
        '''

        print('Setting up lensing reconstruction')
        self.config     = config
        self.lmax    = self.config['lensrec']['lmax'] = max(config['lensrec']['lmaxT'],config['lensrec']['lmaxP'])
        self.Lmax    = self.config['lensrec']['Lmax']
        self.cltype  = self.config['lensrec']['cltype']
        self.cls     = cls
        self.glm = {}
        self.clm = {}

        if self.cltype!='ucmb' and self.cltype!='lcmb' and self.cltype!='grad':
            sys.exit('cltype must be ucmb, lcmb or grad')

        if 'nside' in self.config['lensrec']:
            print("-- Overwrite default nside")
            self.nside = self.config['lensrec']['nside']
            assert self.lmax < 2.0*self.nside, "lmax must be less that 2*nside"
        else:
            self.nside   = utils.get_nside(self.lmax)

        print("-- Nside to project: %d"%self.nside)
        print("-- lmax:%d"%self.lmax)
        print("-- Lmax:%d"%self.Lmax)
        print("-- Using %s cls"%self.cltype)

    def eval(self,qe,alm1all,alm2all,totalcls,u=None,crossilc=False):
        '''
        Compute quadratic estimator

        Parameters
        ----------
        qe : str
          Quadratic estimator type: 'all'/'TTEETE' (TT/EE/TE only)/'TBEB' (TB/EB only)/'TTEETEprf'

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
            if qe == 'TTEETEprf':
                assert u is not None, "Need profile function to compute this estimator"

            if qe == 'all':
                if crossilc:
                    ests = ['TT_GMV', 'TT_GMV', 'EE_GMV', 'TE_GMV', 'ET_GMV', 'TB_GMV', 'BT_GMV', 'EB_GMV', 'BE_GMV']
                    idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                else:
                    ests = ['TT_GMV', 'EE_GMV', 'TE_GMV', 'ET_GMV', 'TB_GMV', 'BT_GMV', 'EB_GMV', 'BE_GMV']
                    idxs = [0, 1, 2, 3, 4, 5, 6, 7]
            elif qe == 'TTEETE':
                if crossilc:
                    ests = ['TT_GMV', 'TT_GMV', 'EE_GMV', 'TE_GMV', 'ET_GMV']
                    idxs = [0, 1, 2, 3, 4]
                else:
                    ests = ['TT_GMV', 'EE_GMV', 'TE_GMV', 'ET_GMV']
                    idxs = [0, 1, 2, 3]
            elif qe == 'TBEB':
                ests = ['TB_GMV', 'BT_GMV', 'EB_GMV', 'BE_GMV']
                if crossilc:
                    idxs = [5, 6, 7, 8]
                else:
                    idxs = [4, 5, 6, 7]
            elif qe == 'TTEETEprf':
                ests = ['TT_GMV_PRF', 'EE_GMV_PRF', 'TE_GMV_PRF','ET_GMV_PRF']
                idxs = [0, 1, 2, 3]
            else:
                print("For GMV, we can only calculate estimators for argument qe 'all', 'TTEETE', 'TBEB', or 'TTEETEprf'")

            print('Running lensing reconstruction')
            retglm = 0
            retclm = 0

            for i, est in enumerate(ests):
                print('Doing estimator: %s'%est)
                idx = idxs[i]
                alm1 = alm1all[:,idx]
                alm2 = alm2all[:,idx]
                q = weights.weights(est, self.cls[self.cltype], self.lmax, u=u, totalcls=totalcls, crossilc=crossilc)
                glmsum = 0
                clmsum = 0

                if est=='TB_GMV' or est=='EB_GMV':
                    print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')

                    # TB first!
                    wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
                    wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

                    walm1 = hp.almxfl(alm1,wX1) # T1
                    walm3 = hp.almxfl(alm1,wX3) # T3
                    walm2 = hp.almxfl(alm2,wY1) # B2

                    SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax)
                    SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax)
                    SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax)

                    SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                    SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                    glm_TB,clm_TB = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                    nrm = 1

                    glm_TB = hp.almxfl(glm_TB,nrm*wP1)
                    clm_TB = hp.almxfl(clm_TB,nrm*wP1)

                    # EB next!
                    wX1,wY1,wP1,sX1,sY1,sP1 = q.w[4][0],q.w[4][1],q.w[4][2],q.s[4][0],q.s[4][1],q.s[4][2]
                    wX3,wY3,wP3,sX3,sY3,sP3 = q.w[6][0],q.w[6][1],q.w[6][2],q.s[6][0],q.s[6][1],q.s[6][2]

                    walm1 = hp.almxfl(alm1,wX1) # E1
                    walm3 = hp.almxfl(alm1,wX3) # E3
                    walm2 = hp.almxfl(alm2,wY1) # B2

                    SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax)
                    SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax)
                    SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax)

                    SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                    SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                    glm_EB,clm_EB = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                    nrm = -1

                    glm_EB = hp.almxfl(glm_EB,nrm*wP1)
                    clm_EB = hp.almxfl(clm_EB,nrm*wP1)

                    # Sum
                    glmsum = glm_TB + glm_EB
                    clmsum = clm_TB + clm_EB

                elif est=='BT_GMV' or est=='BE_GMV':
                    print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')

                    # BT first!
                    wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
                    wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

                    walm1 = hp.almxfl(alm2,wY1) # T1
                    walm3 = hp.almxfl(alm2,wY3) # T3
                    walm2 = hp.almxfl(alm1,wX1) # B2

                    SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax)
                    SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax)
                    SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax)

                    SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                    SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                    glm_BT,clm_BT = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                    nrm = 1

                    glm_BT = hp.almxfl(glm_BT,nrm*wP1)
                    clm_BT = hp.almxfl(clm_BT,nrm*wP1)

                    # BE now...
                    wX1,wY1,wP1,sX1,sY1,sP1 = q.w[4][0],q.w[4][1],q.w[4][2],q.s[4][0],q.s[4][1],q.s[4][2]
                    wX3,wY3,wP3,sX3,sY3,sP3 = q.w[6][0],q.w[6][1],q.w[6][2],q.s[6][0],q.s[6][1],q.s[6][2]

                    walm1 = hp.almxfl(alm2,wY1) # E1
                    walm3 = hp.almxfl(alm2,wY3) # E3
                    walm2 = hp.almxfl(alm1,wX1) # B2

                    SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax)
                    SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax)
                    SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax)

                    SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                    SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                    glm_BE,clm_BE = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                    nrm = -1

                    glm_BE = hp.almxfl(glm_BE,nrm*wP1)
                    clm_BE = hp.almxfl(clm_BE,nrm*wP1)

                    # Sum
                    glmsum = glm_BT + glm_BE
                    clmsum = clm_BT + glm_BE

                else:
                    # More traditional quicklens style calculation
                    for i in range(0,q.ntrm):
                        wX,wY,wP,sX,sY,sP = q.w[i][0],q.w[i][1],q.w[i][2],q.s[i][0],q.s[i][1],q.s[i][2]
                        #print("Computing term %d/%d sj = [%d,%d,%d] of est %s"%(i+1,q.ntrm,sX,sY,sP,est))
                        walm1 = hp.almxfl(alm1,wX)
                        walm2 = hp.almxfl(alm2,wY)

                        # Input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
                        # -----------------------------------------------------------------------------------------------
                        if est[0]=='B':
                            SpX, SmX = hp.alm2map_spin([np.zeros_like(walm1),1j*walm1], self.nside, np.abs(sX), self.lmax)
                        else:
                            SpX, SmX = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, np.abs(sX), self.lmax)

                        X = SpX+1j*SmX # Complex map _{+s}S or _{-s}S

                        if sX<0:
                            X = np.conj(X)*(-1)**(sX)
                        # -----------------------------------------------------------------------------------------------
                        if est[1]=='B':
                            SpY, SmY = hp.alm2map_spin([np.zeros_like(walm2),1j*walm2], self.nside, np.abs(sY), self.lmax)
                        else:
                            SpY, SmY = hp.alm2map_spin([walm2,np.zeros_like(walm2)], self.nside, np.abs(sY), self.lmax)

                        Y = SpY+1j*SmY

                        if sY<0:
                            Y = np.conj(Y)*(-1)**(sY)
                        # -----------------------------------------------------------------------------------------------

                        XY = X*Y

                        if sP<0:
                            XY = np.conj(XY)*(-1)**(sP)

                        glm,clm  = hp.map2alm_spin([XY.real,XY.imag], np.abs(sP), self.Lmax)

                        glmsum += hp.almxfl(glm,0.5*wP)
                        clmsum += hp.almxfl(clm,0.5*wP)

                if est=='TT_GMV' and crossilc is True:
                    nrm = 0.5
                else:
                    nrm = 1
                retglm += nrm*glmsum
                retclm += nrm*clmsum

            self.glm[qe] = retglm
            self.clm[qe] = retclm

        return self.glm[qe], self.clm[qe]

    def get_aresp(self,qe1=None,qe2=None,u=None,filename=None,crossilc=False):
        '''
        Compute analytical response function

        Parameters
        ----------
        filename: string
          Where to save the aresp output to
        qe1: string
          First estimator; if None, assumes it is self.qe
        qe2: string
          Second estimator; if None, assumes it is the same as qe1

        Returns
        ----------
        aresp:
          Analytical response function
        '''
        r = gmv_resp.gmv_resp(self.config,self.cltype,self.totalcls,u=u,save_path=filename,crossilc=crossilc)
        if qe1 is None:
            qe1 = self.qe

        if (qe1 == 'TTEETE' or qe1 == 'TBEB' or qe1 == 'all') and (qe2 is None or qe2 == qe1):
            # Lensing response
            r.calc_tvar()
        elif qe1 == 'TTEETEprf' and (qe2 is None or qe2 == qe1):
            # Source response
            r.calc_tvar_PRF(cross=False)
        elif (qe1=='TTEETE' and qe2=='TTEETEprf') or (qe2=='TTEETE' and qe1=='TTEETEprf'):
            # Cross estimator response of lensing and source
            r.calc_tvar_PRF(cross=True)
        aresp = np.load(filename)

        # Save file has columns L, TTEETE, TBEB, all
        if qe1 == 'TTEETE' or qe1 == 'TTEETEprf':
            aresp = aresp[:,1]
        elif qe1 == 'TBEB':
            aresp = aresp[:,2]
        elif qe1 == 'all':
            aresp = aresp[:,3]
        return aresp

    def harden(self,qe,alm1all,alm2all,totalcls,u,qe_hrd='TTEETEprf',fn_ss=None,fn_es=None,fn_ee=None):
        '''
        Note: We only harden for qe 'all' and 'TTEETE'.
        Getting the hardened plm for TTEETE and then getting the total hardened plm by
        adding it to the unhardened TBEB is equivalent to
        doing the hardening for all in one step (weight is the same in both cases).

        Parameters
        ----------

        Returns
        ----------
        plm :
          Source hardened glm
        resp :
          Response function
        '''
        assert self.qe=='all' or self.qe=='TTEETE', "We only harden for qe 'all' and 'TTEETE', got: %s"%self.qe

        # ee : Response of est*est
        # es : Cross-estimator response of est*src
        # ss : Response of src*src
        ss = self.get_aresp(qe1=qe_hrd,u=u,filename=fn_ss)
        es = self.get_aresp(qe1=qe_hrd,qe2=qe,u=u,filename=fn_es)
        ee = self.get_aresp(qe1=qe,filename=fn_ee)

        plm1,_ = self.eval(qe,alm1all,alm2all,totalcls)
        plm2,_ = self.eval(qe_hrd,alm1all,alm2all,totalcls,u)

        weight = -1*es/ss
        plm    = plm1 + hp.almxfl(plm2, weight)
        resp   = ee + weight*es

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
        self.fl_cut['T'] = flT[:self.lmax+1] if flT is not None else np.ones(self.lmax + 1)
        self.fl_cut['E'] = flP[:self.lmax+1] if flP is not None else np.ones(self.lmax + 1)
        self.fl_cut['B'] = flP[:self.lmax+1] if flP is not None else np.ones(self.lmax + 1)
        assert self.fl_cut['T'].shape[-1] == self.lmax + 1
        assert self.fl_cut['E'].shape[-1] == self.lmax + 1
        assert self.fl_cut['B'].shape[-1] == self.lmax + 1

        self.cls = Cls
        if nside is None:
            nside = utils.get_nside(lmax)
        self.nside=nside
        self.glm = {}
        self.clm = {}

        assert self.lmax < 2.0 * self.nside, "lmax must be less that 2*nside"
        self.gmv = gmv

    @classmethod
    def from_config_yuuki(cls, config, Cls):
        """
        old construction code to work with Yuuki's configuration file
        """

        lmax = config['lensrec']['lmax'] = max(config['lensrec']['lmaxT'],config['lensrec']['lmaxP'])
        Lmax = config['lensrec']['Lmax']

        Cls = Cls[config['lensrec']['cltype']]
        nside = config['lensrec'].get('nside', None)
        return cls(lmax=lmax, nside=nside, Cls=Cls, Lmax=Lmax, )

    @classmethod
    def from_config_srini(cls, dict_lrange, Cls):
        """
        old construction code to work with Srini's configuration file
        """
        lmin = dict_lrange['lmin']
        lmax = max(dict_lrange['lmaxT'], dict_lrange['lmaxP'])
        Lmax = dict_lrange['Lmax']
        nside = None
        return cls(lmax=lmax, nside=nside, Cls=Cls, Lmax=Lmax, )

    @staticmethod
    def alm2map_spin(alm, fell, nside, spin, lmax, mmax=None, g=None):
        """ convert a spin-0 alm into a complex spin field (Q +/- iU): out = Q, +/-U"""
        if spin == 0:
            walm = hp.almxfl(alm, fell)
            # alm2map is recommended over alm2map_spin for spin=0
            if g is None:
                out = hp.alm2map(walm, nside=nside, lmax=lmax, mmax=mmax)
            else:
                out = g.alm2map(walm, lmax=lmax, mmax=mmax)
            return out, 0
        else:
            zero = np.zeros_like(alm)
            _fell = (-1) ** spin * np.conj(fell) if spin < 0 else fell
            _fell *= -1
            if np.all(fell.imag == 0):
                E = hp.almxfl(alm, _fell.real)
                B = zero
            elif np.all(fell.real == 0):
                E = zero
                B = hp.almxfl(alm, _fell.imag)
            else:
                raise ValueError("Fell must be real or imaginary")
            if g is None:
                q, u = hp.alm2map_spin([E, B], nside=nside, spin=np.abs(spin), lmax=lmax, mmax=mmax)
            else:
                q, u = g.alm2map_spin([E, B], spin=np.abs(spin), lmax=lmax, mmax=mmax)
            if spin > 0:
                return q, u
            else:
                return q, -u

    def eval(self, qe, almbar1, almbar2, u=None, g=None, cache=False):
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
        cache: bool=True
            If True, the QE results will be loaded from cache if available.

        Returns
        -------
        glm, clm: tuple of complex array
            Gradient/curl component of the plm
        """
        assert almbar1.shape[-1] == self.size, f"almbar size {almbar1.shape[-1]} don't match lmax {self.lmax})"
        assert almbar2.shape[-1] == self.size, f"almbar size {almbar2.shape[-1]} don't match lmax {self.lmax})"
        if cache and qe in self.glm:
            logger.warning("We've already computed this!")
            return self.glm[qe], self.clm[qe]

        if qe in ['prf']:
            assert u is not None, "Need profile function to compute this estimator"

        q = weights.weights_plus(qe, self.cls, self.lmax, u=u)

        logger.info(f'Running lensing reconstruction: {qe}')

        retglm = 0
        retclm = 0
        if g is not None:
            assert g.nside == self.nside
        assert q.ntrm%2 == 0, f"Number of terms must be even: {q.ntrm}"
        for i in range(0, q.ntrm//2):
            # skipping second half of reducant terms
            wX, wY, wP, sX, sY, sP = q.w[i][0], q.w[i][1], q.w[i][2], q.s[i][0], q.s[i][1], q.s[i][2]

            Xq, Xu = self.alm2map_spin(almbar1, fell=wX, nside=self.nside, spin=sX, lmax=self.lmax, g=g)
            Yq, Yu = self.alm2map_spin(almbar2, fell=wY, nside=self.nside, spin=sY, lmax=self.lmax, g=g)
            XYq = Xq*Yq - Xu*Yu  # XY = X*Y
            XYu = Xq*Yu + Yq*Xu  # XY = X*Y

            if np.all(wP.imag == 0):
                _wP = wP
            elif np.all(wP.real == 0):
                # swap grad/curl mode such that glm is curl and clm is grad
                # wP has an -1j factor, here we move the factor from wP to XY.
                _wP = wP*1j
                XYq, XYu = XYu, -XYq  # XY *=-1j
            else:
                raise ValueError("wP must be real or imaginary")
            if sP < 0:
                # This is for the second half reduncant transform, we normally don't end up here.
                # XY = np.conj(XY) * (-1) ** sP  # because wP has a (-1)**sP factor, here we are canceling it.
                XYq *= (-1) ** sP  # XY = np.conj(XY) * (-1) ** sP
                XYu *= -(-1) ** sP  # XY = np.conj(XY) * (-1) ** sP

            if g is None:
                glm, clm = hp.map2alm_spin([XYq, XYu], np.abs(sP), self.Lmax)
            else:
                glm, clm = g.map2alm_spin([XYq, XYu], spin=np.abs(sP), lmax=self.Lmax, check=False, )
            glm = hp.almxfl(glm, _wP)
            clm = hp.almxfl(clm, _wP)  # for curl est, this will be -grad.

            retglm += glm
            retclm += clm

        self.glm[qe] = retglm
        self.clm[qe] = retclm

        return self.glm[qe], self.clm[qe]

    def get_aresp(self, flX, flY, qe1=None, qe2=None, u=None, fast=False, curl=False):
        """
        Compute analytical response function for 1D filtering

        Parameters
        ----------
        flX, flY
          1D real arrays representing the filter functions for the X and Y fields
        qe1: string
          First estimator
        qe2: string
          Second estimator; if None, assumes it is the same as qe1
        fast: bool=False
            If True, uses the fast response function calculation.
        curl: bool=False
            If True, `qe1` and `qe2` are suffixed with `curl` to compute curl-mode response.
        Returns
        ----------
        aresp:
          Analytical response function
        """
        if qe1 is None:
            assert 0, "qe1 must be defined"

        qeXY = weights.weights_plus(qe1 if not curl else f'{qe1}curl', self.cls, self.lmax, u=u)

        if qe2 is None or qe2 == qe1:
            qeZA = qeXY
        else:
            qeZA = weights.weights_plus(qe2 if not curl else f'{qe2}curl', self.cls, self.lmax, u=u)

        aresp = resp.fill_resp_fullsky(qeXY, qeZA, np.zeros(self.Lmax + 1, dtype=complex), flX, flY, fast=fast)
        return aresp

    def harden(self, qe, almbar1, almbar2, flX, flY, u, qe_hrd='prf', curl=False):
        """
        Get the source hardened glm and the response function.
        Need arguments flX, flY in order to compute the analytical response
        needed for hardening.

        Parameters
        ----------
        qe: string
          Quadratic estimator type, only 'TT' is supported
        flX, flY
          1D real arrays representing the filter functions for the X and Y fields

        Returns
        ----------
        plm :
          Source hardened glm
        resp :
          Response function
        """
        assert qe == 'TT', f"We only harden for qe 'TT', got: {qe}"

        ss = self.get_aresp(flX, flY, qe1=qe_hrd, u=u)
        es = self.get_aresp(flX, flY, qe1=qe_hrd, qe2=qe, u=u, fast=False)
        ee = self.get_aresp(flX, flY, qe1=qe, curl=curl)

        plm1 = self.eval(qe, almbar1, almbar2)[1 if curl else 0]
        plm2 = self.eval(qe_hrd, almbar1, almbar2, u)[0]

        weight = -1 * es / ss
        plm = plm1 + hp.almxfl(plm2, weight)
        R = ee + weight * es
        return plm, R

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

    def get_aresp_gmv(self, fls, qe=None, u=None, fast=False, curl=False, TTprf_s=False, TTprf_x=False):
        """
        Compute analytical response function for GMV (jointly filtered maps)

        Parameters
        ----------
        qe: str
            Quadratic estimator type.
        fls: np.array
            shape: (4, lmax+1), filter functions for TT/EE/BB/TE, i.e., 1/Cltt, 1/Clee, 1/Clbb, 1/Clte.
        u: np.ndarray
            shape (lmax+1, ) profile function for TTprf estimator
        fast: bool=False
            If True, uses the fast response function calculation.
        curl: bool=False
            If True, `qe` is suffixed with `curl` to compute curl-mode response.
        TTprf_s: bool=False
            Perform the source-hardening estimator (only the ss term)
        TTprf_x: bool=False
            Perform the source-hardening estimator (only the se term)
        """
        assert not (TTprf_x and TTprf_s), "TTprf_s and TTprf_x cannot both be True"
        if TTprf_x or TTprf_s:
            assert u is not None, "Need profile function to compute this estimator"

        fl = self.fls2fls_dict(fls)

        s1, s2 = qe[0], qe[1]
        assert s1 in 'TEB' and s2 in 'TEB', f"qe must be one of TEB, got: {qe}"

        if self.gmv:
            keys = list(fl.keys())
        else:
            keys = [f"{s1}{s1}", f"{s2}{s2}"]

        if TTprf_s:
            assert qe=='TT'
            loop1 = loop2 = 'T'
        else:
            loop1 = loop2 = 'TEB'
        if TTprf_s or TTprf_x:
            qeXY = weights.weights_plus('prf', self.cls, self.lmax, u=u)
        else:
            qeXY = weights.weights_plus(qe if not curl else f"{qe}curl", self.cls, self.lmax)

        R = np.zeros(self.Lmax+1, dtype=float)
        for _s1 in loop1:
            _qe1 = s1+_s1
            if _qe1 not in keys:
                continue
            flX = fl[_qe1]*self.fl_cut[s1]
            for _s2 in loop2:
                _qe2 = s2 + _s2
                if _qe2 not in keys:
                    continue
                flY = fl[_qe2]*self.fl_cut[s2]
                if TTprf_s:
                    qeZA = weights.weights_plus('prf', self.cls, self.lmax, u=u)
                else:
                    qeZA = weights.weights_plus(_s1+_s2 if not curl else f"{_s1+_s2}curl", self.cls, self.lmax)
                _R = resp.fill_resp_fullsky(qeXY, qeZA, np.zeros(self.Lmax+1, dtype=complex), flX, flY, fast=fast)
                R += _R
        return R

    def get_harden_weights(self, qe, fls, u, curl=False, fast=False):
        ss = self.get_aresp_gmv(fls, qe='TT', u=u, fast=fast, curl=False, TTprf_s=True)
        es = self.get_aresp_gmv(fls, qe=qe, u=u, fast=fast, curl=curl, TTprf_x=True)
        weight = -1*es/ss
        return weight, es

    def harden_gmv(self, qe, almbar1, almbar2, fls, u, curl=False, fast=False):
        """
        Get the source hardened glm and the response function.
        Need arguments flX, flY in order to compute the analytical response
        needed for hardening.

        Parameters
        ----------
        qe: str
            Quadratic estimator type, only 'TT' is supported
        almbar1,almbar2: complex array healpy alm
            First and second filtered alm
        fls: np.ndarray
            shape: (4, lmax+1), filter functions for TT/EE/BB/TE, i.e., 1/Cltt, 1/Clee, 1/Clbb, 1/Clte.
        u: np.ndarray
            shape (lmax+1, ) profile function for TTprf estimator
        fast: bool=False
            If True, uses the fast response function calculation.
        curl: bool=False
            If True, `qe1` is suffixed with `curl` to compute curl-mode response.

        Returns
        -------
        plm: complex array healpy alm
          Source hardened glm
        resp: np.ndarray
          Response function
        """
        assert qe.endswith('prf') is False, "'prf' should be stripped before doing hardening"
        if not self.gmv:
            assert qe == 'TT', f"We only harden for 'TT' for SQE, got: {qe}"

        weight, es = self.get_harden_weights(qe, fls, u, curl=curl, fast=fast)
        ee = self.get_aresp_gmv(fls, qe=qe, fast=fast, curl=curl)

        plm_len = self.eval(qe, almbar1, almbar2)[1 if curl else 0]
        plm_src = self.eval('prf', almbar1, almbar2, u=u)[0]

        plm = plm_len + hp.almxfl(plm_src, weight)
        R = ee + weight * es
        return plm, R

    def rec_and_resp(self, qe, almbars1, almbars2, fls, u=None, g=None, fast=False):
        """
        compute lensing reconstruction for grad and curl modes, return also the analytical response functions.

        Parameters
        ----------
        qe: str
            Quadratic estimator type, e.g., 'TT','TTprf'
        almbars1, almbars2: complex arrays
            First and second filtered alms, shape (3, nalm)
        fls: np.ndarray
            shape: (4, lmax+1), filter functions for TT/EE/BB/TE
        u: np.ndarray=None
            profile function for TTprf estimator
        g: Geometry=None
            Geometry instance defined within declination range.
        fast: bool=False
            If True, uses the fast response function calculation.

        Returns
        -------
        [glm, clm]: list of complex array
            Gradient/curl component of the plm
        [aresp_g, aresp_c]: list of np.ndarray
            Analytical response function for grad/curl mode
        """

        i1 = 'teb'.index(qe[0].lower())
        i2 = 'teb'.index(qe[1].lower())

        if not qe.endswith('prf'):
            glm, clm = self.eval(qe, almbars1[i1], almbars2[i2], g=g)
            aresp_g = self.get_aresp_gmv(fls, qe, fast=fast)
            aresp_c = self.get_aresp_gmv(fls, qe, fast=fast, curl=True)
        else:
            glm, aresp_g = self.harden_gmv(qe.removesuffix("prf"), almbars1[i1], almbars2[i2], fls,
                                           u=u, curl=False, fast=fast)
            clm, aresp_c = self.harden_gmv(qe.removesuffix("prf"), almbars1[i1], almbars2[i2], fls,
                                           u=u, curl=True, fast=fast)
        return [glm, clm], [aresp_g, aresp_c]
