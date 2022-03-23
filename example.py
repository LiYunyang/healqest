# This just tests all the estimators
import sys
import numpy as np
import healpy as hp
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/')
import utils
import weights
import qest
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/Quicklens-with-fixes/')
import quicklens as ql
import camb

####################################
est     = str(sys.argv[1])
lmax    = 1000 # for quick testing
fwhm    = 1
nlev_t  = 5
nlev_p  = 5
dir_out = '/lcrc/project/SPT3G/users/ac.yomori/scratch/qetest/'
clfile  = '/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/mdpl2_lensedCls.dat'
####################################

pars=camb.read_ini('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/mdpl2_params.ini')
pars.max_l=1500
results = camb.get_results(pars)
sltt,slee,slbb,slte = results.get_cmb_power_spectra(pars,lmax=lmax, CMB_unit='muK',raw_cl=True)['lensed_scalar'].T

# /lensed_cmb_mdpl2phiG/mdpl2_lensedTQU1_mdpl2phiG map
tlm,elm,blm = hp.read_alm('/lcrc/project/SPT3G/users/ac.yomori/scratch/qetest/inputcmb.alm',hdu=[1,2,3])
'''
bl         = hp.gauss_beam(fwhm=fwhm*0.00029088,lmax=lmax)
nltt       = (np.pi/180./60.*nlev_t)**2 / bl**2
nlee=nlbb  = (np.pi/180./60.*nlev_p)**2 / bl**2
zero       = np.zeros_like(nltt)

nlmt,nlme,nlmb = hp.synalm([nltt,nlee,nlee,zero])
hp.write_alm('/lcrc/project/SPT3G/users/ac.yomori/scratch/qetest/noiseteb.alm',[nlmt,nlme,nlmb],overwrite=True)
'''
nlmt,nlme,nlmb = hp.read_alm('/lcrc/project/SPT3G/users/ac.yomori/scratch/qetest/noiseteb.alm',hdu=[1,2,3])
nl = hp.alm2cl([nlmt,nlme,nlmb])
nltt=nl[0]
nlee=nl[1]
nlbb=nl[2]

# signal+noise spectra
cltt       = sltt + nltt
clee       = slee + nlee
clbb       = slbb + nlbb

# filter functions
flt        = np.zeros( lmax+1 ); flt[2:] = 1./cltt[2:]
fle        = np.zeros( lmax+1 ); fle[2:] = 1./clee[2:]
flb        = np.zeros( lmax+1 ); flb[2:] = 1./clbb[2:]

if est[0] == 'T': almbar1 = hp.almxfl(tlm+nlmt,flt); flm1= flt
if est[0] == 'E': almbar1 = hp.almxfl(elm+nlme,fle); flm1= fle
if est[0] == 'B': almbar1 = hp.almxfl(blm+nlmb,flb); flm1= flb

if est[1] == 'T': almbar2 = hp.almxfl(tlm+nlmt,flt); flm2= flt
if est[1] == 'E': almbar2 = hp.almxfl(elm+nlme,fle); flm2= fle
if est[1] == 'B': almbar2 = hp.almxfl(blm+nlmb,flb); flm2= flb

# healqest
glm,clm = qest.qest(est,lmax,clfile,almbar1,almbar2)
np.save(dir_out+'/plm_%s_healqest.npy'%est,glm)
np.save(dir_out+'/clm_%s_healqest.npy'%est,clm)


# quicklens with fixes
if est=='TT': Q = ql.qest.lens.phi_TT(sltt[:lmax+1])
if est=='EE': Q = ql.qest.lens.phi_EE(slee[:lmax+1])
if est=='TE': Q = ql.qest.lens.phi_TE(slte[:lmax+1])
if est=='TB': Q = ql.qest.lens.phi_TB(slte[:lmax+1])
if est=='EB': Q = ql.qest.lens.phi_EB(slee[:lmax+1])
Q.lmax      = lmax
#vlm         = Q.eval_fullsky(almbar1,almbar2)
#glm,clm     = ql.shts.util.vlm2alm(vlm)
#np.save(dir_out+'/plm_%s_quicklensfixed.npy'%est,glm)

resp = Q.fill_resp(Q, np.zeros(lmax+1, dtype=np.complex), flm1,flm2)
l=np.arange(lmax+1)
np.savetxt(dir_out+'resp_%s_plm.dat'%est,np.c_[l,resp.real])



