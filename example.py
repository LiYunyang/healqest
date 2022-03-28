# This just tests all the estimators
import sys
import numpy as np
import healpy as hp
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/')
import utils
import weights
import qest
import camb
from pathlib import Path

####################################
est       = str(sys.argv[1]) # TT/EE/TE/TB/EB
lmax      = 1000             # For quick testing
fwhm      = 1
nlev_t    = 5
nlev_p    = 5
cambini   = 'camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini'
clfile    = 'camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
file_alm  = './llcdm/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed1_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap_lmax2048.alm'
dir_out   = './testout'
####################################

# Run camb to get theory cls
# This might be slow for dpending on the compiler used.
# If so, its probably better to run it once and save the cls so that it can be loaded the next time.
#pars       = camb.read_ini(cambini)

#Set up a new set of parameters for CAMB (using random cosmology to speed up calculation)
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);
results    = camb.get_results(pars)
sltt,slee,slbb,slte = results.get_cmb_power_spectra(pars,lmax=lmax, CMB_unit='muK',raw_cl=True)['lensed_scalar'].T

# Load inputs fullsky noiseless alms
tlm,elm,blm = hp.read_alm(file_alm,hdu=[1,2,3])
tlm         = utils.reduce_lmax(tlm,lmax=lmax)
elm         = utils.reduce_lmax(elm,lmax=lmax)
blm         = utils.reduce_lmax(blm,lmax=lmax)

# Create noise spectra
bl          = hp.gauss_beam(fwhm=fwhm*0.00029088,lmax=lmax)
nltt        = (np.pi/180./60.*nlev_t)**2 / bl**2
nlee=nlbb   = (np.pi/180./60.*nlev_p)**2 / bl**2

nlmt,nlme,nlmb = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True) 

# signal+noise spectra
cltt        = sltt + nltt
clee        = slee + nlee
clbb        = slbb + nlbb

# Create 1/Nl filters
flt         = np.zeros( lmax+1 ); flt[2:] = 1./cltt[2:]
fle         = np.zeros( lmax+1 ); fle[2:] = 1./clee[2:]
flb         = np.zeros( lmax+1 ); flb[2:] = 1./clbb[2:]

if est[0] == 'T': almbar1 = hp.almxfl(tlm+nlmt,flt); flm1= flt
if est[0] == 'E': almbar1 = hp.almxfl(elm+nlme,fle); flm1= fle
if est[0] == 'B': almbar1 = hp.almxfl(blm+nlmb,flb); flm1= flb

if est[1] == 'T': almbar2 = hp.almxfl(tlm+nlmt,flt); flm2= flt
if est[1] == 'E': almbar2 = hp.almxfl(elm+nlme,fle); flm2= fle
if est[1] == 'B': almbar2 = hp.almxfl(blm+nlmb,flb); flm2= flb

# run healqest
glm,clm = qest.qest(est,lmax,clfile,almbar1,almbar2)

#save plm and clm
Path(dir_out).mkdir(parents=True, exist_ok=True)
np.save(dir_out+'/plm_%s_healqest.npy'%est,glm)
np.save(dir_out+'/clm_%s_healqest.npy'%est,clm)

