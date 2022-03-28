#test_fullsky.py

import sys
sys.path.insert(0, '/project2/chihway/yuuki/repo/lensing/')
import healpy as hp
import numpy as np
#import quicklens as ql
import utils
import weights
import qest
import quicklens as ql

def zeropad(cl):
    cl = np.insert(cl,0,0)
    cl = np.insert(cl,0,0)
    return cl

seed1 = int(sys.argv[1]) 
seed2 = int(sys.argv[2]) 

lmax  = 4000

clfile = '/project2/chihway/yuuki/repo/sptsz_mapmaking/data/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
ell,sltt,slee,slbb,slte=np.loadtxt('/project2/chihway/yuuki/repo/sptsz_mapmaking/data/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',unpack=True)
sltt=sltt/ell/(ell+1)*2*np.pi; sltt=zeropad(sltt)

fwhmT    = 1.
nlevT    = 5.0
blT      = hp.gauss_beam(fwhm=fwhmT*0.000290888,lmax=lmax)
nltt     = (np.pi/180./60.*nlevT)**2 / blT**2

zlmT     = 1.0/(sltt[:lmax+1]+nltt[:lmax+1]); zlmT[:100]=0

t1    = hp.read_map('/project2/chihway/yuuki/sptsz2500d_v5/lensed_TQU1phi1/lensedTQU1phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed%d_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_map.fits'%seed1)
nmap1 = hp.synfast(nltt,8192)
almT1 = hp.map2alm(t1+nmap1,lmax=lmax)
del t1,nmap1

t2    = hp.read_map('/project2/chihway/yuuki/sptsz2500d_v5/lensed_TQU1phi1/lensedTQU1phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed%d_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_map.fits'%seed2)
nmap2 = hp.synfast(nltt,8192)
almT2 = hp.map2alm(t2+nmap2,lmax=lmax)
del t2,nmap2

t3    = hp.read_map('/project2/chihway/yuuki/sptsz2500d_v5/lensed_TQU2phi1/lensedTQU2phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed%d_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_map.fits'%seed1)
nmap3 = hp.synfast(nltt,8192)
almT3 = hp.map2alm(t3+nmap3,lmax=lmax)
del t3,nmap3

ell,em=hp.Alm.getlm(lmax)
tlmbar1   = zlmT[ell]*almT1
tlmbar2   = zlmT[ell]*almT2
tlmbar3   = zlmT[ell]*almT3

Q           = ql.qest.lens.phi_TT(sltt[:lmax+1])
Q.lmax      = lmax
rlpp        = np.zeros(lmax+1,dtype=np.complex_)
rlpp        = Q.fill_resp(Q,rlpp,zlmT,zlmT).real
np.savetxt('/scratch/midway2/yomori/tmp2/rlpp_TT.dat',rlpp)


glmT1T1,clmT1T1 = qest.qest('TT',lmax,clfile,tlmbar1,tlmbar1) #reconstruct TT lensing 
np.save('/scratch/midway2/yomori/tmp2/plmxx_%d_%d.alm'%(seed1,seed1),glmT1T1)

glmT1T2,clmT1T2 = qest.qest('TT',lmax,clfile,tlmbar1,tlmbar2) #reconstruct TT lensing 
np.save('/scratch/midway2/yomori/tmp2/plmxy_%d_%d.alm'%(seed1,seed2),glmT1T2)

glmT1T3,clmT1T3 = qest.qest('TT',lmax,clfile,tlmbar1,tlmbar3) #reconstruct TT lensing 
np.save('/scratch/midway2/yomori/tmp2/plmab_%d_%d.alm'%(seed1,seed1),glmT1T3)

