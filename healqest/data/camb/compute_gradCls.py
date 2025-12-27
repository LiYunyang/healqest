import os,sys,camb

cambini = str(sys.argv[1])

# This camb file is using very high resolution seetings so it is extremely slow to compute
#cambini = '/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/pipeline/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini'#str(sys.argv[1])

pars                 = camb.read_ini(cambini)
results              = camb.get_results(pars)
sltt,slee,slbb,slte  = results.get_cmb_power_spectra(pars, CMB_unit='muK',raw_cl=False)['lensed_scalar'].T

TgT,EgE,BgB,PPt,TgE,TPt,gT2,gTgT = results.get_lensed_gradient_cls(CMB_unit='muK', raw_cl=False, clpp=None).T

l=np.arange(len(sltt))
dir   = str(pathlib.Path(cambini).parent)+'/'
fname1 = pathlib.Path('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/pipeline/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini').stem[:-6]+'lensedCls_gradlensedCls.dat'
fname2 = pathlib.Path('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/pipeline/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini').stem[:-6]+'gradlensedCls.dat'

np.savetxt(dir+fname1, np.c_[l,sltt,slee,slbb,slte,TgT,EgE,BgB,PPt,TgE,TPt,gT2,gTgT][2:,:], header='[0]ell [1]TT [2]EE [3]BB [4]TE [5]TgradT [6]EgradE [7]BgradB [8]PPt [9]TgradE [10]TPt [11](gradT)^2 [12]gradTgradT')
np.savetxt(dir+fname2, np.c_[l,TgT,EgE,BgB,TgE][2:,:], header='[0]ell [1]TgradT [2]EgradE [3]BgradB [4]TgradE')
