import sys
sys.path.insert(0, '/project2/chihway/yuuki/repo/lensing/')
import healpy as hp
import numpy as np
import quicklens as ql
import utils
import weights
import qest

seed1 = int(sys.argv[1])
seed2 = int(sys.argv[2])

#seed1=2
#seed2=2

'''
for seed in range(28,50):
    t,q,u=hp.read_map('lensedTQU1phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed%d_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_map.fits'%seed,field=[0,1,2])
    almT,almE,almB=hp.map2alm([t,q,u],lmax=4096)
    hp.write_alm('almT_seed%d.alm'%seed,almT) 
    hp.write_alm('almE_seed%d.alm'%seed,almE) 
    hp.write_alm('almB_seed%d.alm'%seed,almB) 
'''
lmax      = 4096
Lmax      = 4096
fwhm      = 1.
nlevT     = 2.2
nlevP     = 2.2*2**0.5
bl        = hp.gauss_beam(fwhm=fwhm*0.000290888,lmax=lmax)
nltt      = (np.pi/180./60.*nlevT)**2/bl**2
nlee=nlbb = (np.pi/180./60.*nlevP)**2/bl**2

almT1  = hp.read_alm('/project2/chihway/yuuki/3gtests/alms/almT_seed%d.alm'%seed1)
almE1  = hp.read_alm('/project2/chihway/yuuki/3gtests/alms/almE_seed%d.alm'%seed1)
almB1  = hp.read_alm('/project2/chihway/yuuki/3gtests/alms/almB_seed%d.alm'%seed1)

t,q,u = hp.alm2map([almT1,almE1,almB1],2048,verbose=False)
nmapt = hp.synfast(nltt,2048,verbose=False)
nmapq = hp.synfast(nlee,2048,verbose=False)
nmapu = hp.synfast(nlbb,2048,verbose=False)

almT1,almE1,almB1 = hp.map2alm([t+nmapt,q+nmapq,u+nmapu],lmax=4096)

almT2  = hp.read_alm('/project2/chihway/yuuki/3gtests/alms/almT_seed%d.alm'%seed2)
almE2  = hp.read_alm('/project2/chihway/yuuki/3gtests/alms/almE_seed%d.alm'%seed2)
almB2  = hp.read_alm('/project2/chihway/yuuki/3gtests/alms/almB_seed%d.alm'%seed2)

t,q,u = hp.alm2map([almT2,almE2,almB2],2048,verbose=False)
nmapt = hp.synfast(nltt,2048,verbose=False)
nmapq = hp.synfast(nlee,2048,verbose=False)
nmapu = hp.synfast(nlbb,2048,verbose=False)

almT2,almE2,almB2 = hp.map2alm([t+nmapt,q+nmapq,u+nmapu],lmax=4096)

# ----------- reconstructing lens ---------------

clfile = '/project2/chihway/yuuki/repo/sptsz_mapmaking/data/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
ell,sltt,slee,slbb,slte = utils.get_lensedcls(clfile,lmax)

zlmT=1.0/(sltt[:lmax+1]+nltt[:lmax+1]); zlmT[:100]=0
zlmE=1.0/(slee[:lmax+1]+nlee[:lmax+1]); zlmE[:100]=0
zlmB=1.0/(slbb[:lmax+1]+nlee[:lmax+1]); zlmB[:100]=0

tlmbar1=hp.almxfl(almT1,zlmT)
elmbar1=hp.almxfl(almE1,zlmE)
blmbar1=hp.almxfl(almB1,zlmB)

tlmbar2=hp.almxfl(almT2,zlmT)
elmbar2=hp.almxfl(almE2,zlmE)
blmbar2=hp.almxfl(almB2,zlmB)

print("Computing TT response ..."),
Q           = ql.qest.lens.phi_TT(sltt[:lmax+1])
Q.lmax      = lmax
rlppTT      = np.zeros(lmax+1,dtype=np.complex_)
rlppTT      = Q.fill_resp(Q,rlppTT,zlmT,zlmT).real
rlppTT[:3]  = 1e30
np.savetxt('/project2/chihway/yuuki/3gtests/plms/rlppTT.dat',rlppTT)
print("done")

print("Computing TE response ..."),
Q           = ql.qest.lens.phi_TE(slte[:lmax+1])
Q.lmax      = lmax
rlppTE      = np.zeros(lmax+1,dtype=np.complex_)
rlppTE      = Q.fill_resp(Q,rlppTE,zlmT,zlmE).real
rlppTE[:3]  = 1e30
np.savetxt('/project2/chihway/yuuki/3gtests/plms/rlppTE.dat',rlppTE)
print("done")

print("Computing qest")
glmT1T1,clmT1T1 = qest.qest('TT',lmax,clfile,tlmbar1,tlmbar2) #reconstruct TT lensing 
glmT1E1,clmT1E1 = qest.qest('TE',lmax,clfile,tlmbar1,elmbar2) #reconstruct TT lensing 
#glmTE,clmTE = qest.qest('TE',lmax,almT,almE) #reconstruct TT lensing 
#glmEE,clmEE = qest.qest('EE',lmax,almE,almE) #reconstruct TT lensing 
hp.write_alm('/project2/chihway/yuuki/3gtests/plms/plmTT_%d_%d.alm'%(seed1,seed2),glmT1T1)
hp.write_alm('/project2/chihway/yuuki/3gtests/plms/plmTE_%d_%d.alm'%(seed1,seed2),glmT1E1)
#------------- plot ----------------
ell,emm=hp.Alm.getlm(lmax)
clkkT1T1=hp.alm2cl(0.5*ell*(ell+1)*glmT1T1,0.5*ell*(ell+1)*glmT1T1)/rlppTT**2
plt.clf()
l=np.arange(lmax+1)
t=np.loadtxt('/project2/chihway/yuuki/repo/sptsz_mapmaking/data/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat')
plt.loglog(t[:,0],t[:,5]*2*np.pi/4.,'k-',lw=1.25)
plt.loglog(l,clkkT1T1-(l*(l+1))**2/4/rlppTT,color='firebrick',alpha=0.5)
plt.loglog(l,(l*(l+1))**2/4/rlppTT,'gray')
plt.xlim(5,4000)
plt.ylim(1e-12,1e-3)