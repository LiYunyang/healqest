import utils
import weights
import qest
import sys

seed = int(sys.argv[1])

'''
for seed in range(11,50):
    t,q,u=hp.read_map('lensedTQU1phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed%d_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_map.fits'%seed,field=[0,1,2])
    almT,almE,almB=hp.map2alm([t,q,u],lmax=4096)
    hp.write_alm('almT_seed%d.alm'%seed,almT) 
    hp.write_alm('almE_seed%d.alm'%seed,almE) 
    hp.write_alm('almB_seed%d.alm'%seed,almB) 
'''
lmax      = 4096
fwhm      = 1.
nlevT     = 2.2
nlevP     = 2.2*2**0.5
bl        = hp.gauss_beam(fwhm=fwhm*0.000290888,lmax=lmax)
nltt      = (np.pi/180./60.*nlevT)**2/bl**2
nlee=nlbb = (np.pi/180./60.*nlevP)**2/bl**2

almT  = hp.read_alm('/project2/chihway/yuuki/3gtests/alms/almT_seed%d.alm'%seed)
almE  = hp.read_alm('/project2/chihway/yuuki/3gtests/alms/almE_seed%d.alm'%seed)
almB  = hp.read_alm('/project2/chihway/yuuki/3gtests/alms/almB_seed%d.alm'%seed)

t,q,u = hp.alm2map([almT,almE,almB],2048)
nmapt = hp.synfast(nltt,2048)
nmapq = hp.synfast(nlee,2048)
nmapu = hp.synfast(nlbb,2048)

almT,almE,almB = hp.map2alm([t+nmapt,q+nmapq,u+nmapu],lmax=4096)

# ----------- reconstructing lens ---------------
lmax   = 4096
clfile = '/project2/chihway/yuuki/repo/sptsz_mapmaking/data/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
glmTT,clmTT = qest.qest('TT',lmax,clfile,almT,almT) #reconstruct TT lensing 
#glmTE,clmTE = qest.qest('TE',lmax,almT,almE) #reconstruct TT lensing 
#glmEE,clmEE = qest.qest('EE',lmax,almE,almE) #reconstruct TT lensing 


