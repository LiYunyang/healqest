import sys
import yaml
import argparse
import pathlib
import healpy as hp
import numpy  as np

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml' , default=None, type=str, help='yaml file with all the configurations')
parser.add_argument('ilctype'   , default=None, type=str, help='cmbmv/ynull')
parser.add_argument('cmbset'    , default=None, type=int, help='cmbset')
parser.add_argument('seed'      , default=None, type=int, help='seed')
args = parser.parse_args()

ilctype = args.ilctype
seed    = args.seed
cmbset  = args.cmbset

params  = yaml.safe_load(open(args.file_yaml))
dir_tmp = params['dirs']['dir_tmp']
suff    = params['runtime']['suff']

pathlib.Path(dir_tmp+'/gaussmaps/').mkdir(parents=True, exist_ok=True)
pathlib.Path(dir_tmp+'/gaussmaps/for_mf_resp/').mkdir(parents=True, exist_ok=True)

mask = params['masks']['dir_mask']+params['masks']['Tmaska']
m    = hp.read_map(mask)

lmax=5100

if seed==0:
    comps     = {#'tot'  : {'name':'lcmbNG_lcibNG_ltszNGbahamas80scaled1.000_lkszNGbahamas80_lradNG_noise', 'pol': True},
                 #'totfg': {'name':'lcibNG_ltszNGbahamas80scaled1.000_lkszNGbahamas80_lradNG', 'pol': True},
                 'rad'  : {'name':'lradNG', 'pol': True},
                 #'cmb'  : {'name':'lcmbNG', 'pol': True},
                 'cib'  : {'name':'lcibNG', 'pol': False},
                 'tsz'  : {'name':'ltszNGbahamas80scaled1.000', 'pol': False},
                 'ksz'  : {'name':'lkszNGbahamas80', 'pol': False},
                }

    if cmbset==1:
        #Save alms of real maps
        tot=0
        for comp in (comps):
            d = hp.read_map(dir_tmp+'/ilc/mdpl2_%s_%s_%s_uk_inpainted.fits'%(suff,ilctype,comps[comp]['name']) ,field=[0,1,2])
            alm = hp.map2alm(d*m,lmax=lmax,use_pixel_weights=True)
            hp.write_alm(dir_tmp+'/gaussmaps/%s_%s_%d.alm'%(comp,ilctype,seed),alm,overwrite=True) # say data is cmbset=1
            tot+=d

        alm = hp.map2alm( [tot[0]*m, tot[1]*m, tot[2]*m],lmax=lmax,use_pixel_weights=True)
        hp.write_alm(dir_tmp+'/gaussmaps/%s_%s_%d.alm'%('totfg',ilctype,seed),alm,overwrite=True)

        cmb = hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.7/cmb/len/mdpl2_lensedTQU1_mdpl2phiNG_seed%d_lmax16000_nside8192_interp1.6_method1_pol_1_lensedmap.fits'%(seed+10000),field=[0,1,2])
        alm = hp.map2alm( [cmb[0], cmb[1], cmb[2]],lmax=lmax,use_pixel_weights=True)
        hp.write_alm(dir_tmp+'/gaussmaps/cmb%d_%s_%s_%d.alm'%(1,ilctype,'cmb',seed),alm,overwrite=True)

        cls_noise = np.loadtxt('./noise/cls_noise_%s.dat'%ilctype)[:lmax+1,1:5]
        alms      = hp.synalm((cls_noise).T,new=True)
        nmap      = hp.alm2map(alms,8192)

        cmb[0]+=(tot[0]*m+nmap[0]) # fullsky cmb + masked fg + fullsky noise
        cmb[1]+=(tot[1]*m+nmap[1])
        cmb[2]+=(tot[2]*m+nmap[2])
        alm  = hp.map2alm( [cmb[0], cmb[1], cmb[2]],lmax=lmax,use_pixel_weights=True)

        hp.write_alm(dir_tmp+'/gaussmaps/cmb%d_%s_phiNG_tszG_kszG_cibG_radG_%d.alm'%(1,ilctype,seed),alm,overwrite=True)

else:
    # gaussian totfgs+noise
    np.random.seed(seed)
    cls_noise = np.loadtxt('./noise/cls_noise_%s.dat'%ilctype)[:lmax+1,1:5]
    cls_totfg = np.loadtxt('./fgcls/cls_totfg_%s.dat'%ilctype)[:lmax+1,1:5]

    alms   = hp.synalm((cls_totfg).T,new=True)
    fgmap  = hp.alm2map(alms,8192)

    alms   = hp.synalm((cls_noise).T,new=True)
    nmap   = hp.alm2map(alms,8192)

    # Load cmb maps
    print("----- Load lensed CMB map")
    cmb     = hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.7/cmb/len/mdpl2_lensedTQU%d_mdpl2phiG_seed%d_lmax16000_nside8192_interp1.6_method1_pol_1_lensedmap.fits'%(cmbset,seed),field=[0,1,2])

    totT = cmb[0]+fgmap[0]*m+nmap[0]
    totQ = cmb[1]+fgmap[1]*m+nmap[1]
    totU = cmb[2]+fgmap[2]*m+nmap[2]

    # Save maps
    alm=hp.map2alm([totT,totQ,totU],lmax=lmax,use_pixel_weights=True)
    hp.write_alm(dir_tmp+'/gaussmaps/cmb%d_%s_phiG_tszG_kszG_cibG_radG_%d.alm'%(cmbset,ilctype,seed),[alm[0],alm[1],alm[2]],overwrite=True)
