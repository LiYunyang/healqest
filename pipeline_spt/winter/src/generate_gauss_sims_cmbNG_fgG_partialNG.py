# Generate many realisations of CMB NG + totfg G
# CMB NG is fixed but totFG is varied
# comp is the componnet you want to replace with NG realisation
import yaml
import argparse
import pathlib
import healpy as hp
import numpy  as np
import sys,os
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/skymapkit/src/utils/')
import utils
import subprocess 

os.environ['HEALPIX'] = "/lcrc/project/SPT3G/users/ac.yomori/envs/analysis/Healpix_3.80/"
spice='/lcrc/project/SPT3G/users/ac.yomori/packages/PolSpice_v03-07-03/bin/spice'


parser = argparse.ArgumentParser()
parser.add_argument('file_yaml' , default=None, type=str, help='yaml file with all the configurations')
parser.add_argument('ilctype'   , default=None, type=str, help='cmbmv/ynull')
parser.add_argument('comp'      , default=None, type=str, help='comp')
parser.add_argument('seed'      , default=None, type=int, help='seed')
args = parser.parse_args()


ilctype = args.ilctype
seed    = args.seed
comp    = args.comp
cmbset  = 1

params  = yaml.safe_load(open(args.file_yaml))
dir_tmp = params['dirs']['dir_tmp']
suff    = params['runtime']['suff']

pathlib.Path(dir_tmp+'/gaussmaps_cmbNG_fgG_partialNG/').mkdir(parents=True, exist_ok=True)

mask = params['masks']['dir_mask']+params['masks']['Tmaska']
m    = hp.read_map(mask)


comps     = {
            'all' : {'name':'lcibNG_ltszNGbahamas80scaled1.000_lkszNGbahamas80_lradNG', 'pol': True},
            'rad'   : {'name':'lradNG', 'pol': True},
            'cib'   : {'name':'lcibNG', 'pol': False},
            'tsz'   : {'name':'ltszNGbahamas80scaled1.000', 'pol': False},
            'ksz'   : {'name':'lkszNGbahamas80', 'pol': False},
            'cibtsz': {'name':'lcibNG_ltszNGbahamas80scaled1.000', 'pol': False},
            }
        

if comp=='none': fid = 'tszG_kszG_cibG_radG'
if comp=='tsz' : fid = 'tszNG_kszG_cibG_radG'
if comp=='ksz' : fid = 'tszG_kszNG_cibG_radG'
if comp=='cib' : fid = 'tszG_kszG_cibNG_radG'
if comp=='rad' : fid = 'tszG_kszG_cibG_radNG'
if comp=='all' : fid = 'tszNG_kszNG_cibNG_radNG'



np.random.seed(seed)
cls_noise = np.loadtxt('./noise/cls_noise_%s.dat'%ilctype)[:5101,1:5]

alms   = hp.synalm((cls_noise).T,new=True)
nmapG  = hp.alm2map(alms,8192)

cls_totfg = np.loadtxt('./fgcls/cls_totfg_%s.dat'%ilctype)[:5101,1:5]


'''
print(comp)
if comp == 'rad' or comp == 'totfg':
    # Need to load all the polarisations as well
    cls_comp = np.loadtxt('/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/lensedfg/cls_spt3g_%s_%s_masked8sig.dat'%(ilctype,comp))[:5101,1:5]
     
elif comp == 'none':
    print('assuming all components to be Gaussian')
    cls_comp = cls_totfg*0

else:
    cls_comp = cls_totfg*0
    cls_comp[:,0] = np.loadtxt('/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/lensedfg/cls_spt3g_%s_%s_masked8sig.dat'%(ilctype,comp))[:5101,1]


alms   = hp.synalm((cls_totfg-cls_comp).T,new=True)
fgmapG = hp.alm2map(alms,8192)
'''

cmb = hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.7/cmb/len/mdpl2_lensedTQU1_mdpl2phiNG_seed%d_lmax16000_nside8192_interp1.6_method1_pol_1_lensedmap.fits'%(seed+10000),field=[0,1,2])

# Adding Gaussian component
if comp == 'none':
    # Not adding anything since everythign is gaussian
    fgmapNG = nmapG*0

    alms    = hp.synalm((cls_totfg).T,new=True)
    fgmapG  = hp.alm2map(alms,8192)

else:
    # Adding nonGaussian foregrounds

    fname   = dir_tmp+'/ilc/mdpl2_cmbs4w_%s_%s_uk_inpainted.fits'%(ilctype,comps[comp]['name'])
    
    '''
    file_maskT = params['masks']['dir_mask']+params['masks']['Tmaska']

    subprocess.call(['ls',fname])
    subprocess.call(['ls',file_maskT])

    if os.path.exists(fname)==False or os.path.exists(file_maskT)==False:
        sys.exit("file does not exist")
    else:
        print('found both files')
        
    utils.run_polspice(spice, mapfile     = fname,
                              mapfile2    = fname,
                              weightfile  = file_maskT,
                              weightfile2 = file_maskT,
                              clfile      = './fgcls/cls_%s_spt3g_%s.dat'%(comp,ilctype),
                              nlmax=6500, apodizesigma=170, thetamax=180, beam=0, beam2=0, subav='YES', polarization='YES', decouple='YES'
                       )
    '''
    fgmapNG = hp.read_map(fname ,field=[0,1,2])
    
    if comp=='all': comp = 'totfg'

    cls     = np.loadtxt('./fgcls/cls_%s_cmbs4w_%s.dat'%(comp,ilctype))[:5101,1:5]

    if comp == 'ksz' or comp=='tsz' or comp=='cib':
        cls[:,1:]=0 # set all polarisation to 0

    alms    = hp.synalm((cls_totfg-cls).T,new=True)
    fgmapG  = hp.alm2map(alms,8192)

totT = cmb[0] + (fgmapG[0]+fgmapNG[0])*m + nmapG[0]
totQ = cmb[1] + (fgmapG[1]+fgmapNG[1])*m + nmapG[1]
totU = cmb[2] + (fgmapG[2]+fgmapNG[2])*m + nmapG[2]

# Save maps
alm=hp.map2alm([totT,totQ,totU],lmax=5100)
hp.write_alm(dir_tmp+'/gaussmaps_cmbNG_fgG_partialNG/cmb%d_%s_phiNG_%s_0_gauss%d.alm'%(cmbset,ilctype,fid,seed),[alm[0],alm[1],alm[2]],overwrite=True)
