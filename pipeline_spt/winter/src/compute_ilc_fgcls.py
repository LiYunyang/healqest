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
parser.add_argument('idx'       , default=None, type=int, help='idx')
#parser.add_argument('seed'     , default=None, type=int, help='seed')
args = parser.parse_args()


ilctype = args.ilctype
idx     = args.idx
#comp    = args.comp
cmbset  = 1

params  = yaml.safe_load(open(args.file_yaml))
dir_tmp = params['dirs']['dir_tmp']
suff    = params['runtime']['suff']

#pathlib.Path(dir_tmp+'/gaussmaps_cmbNG_fgG_partialNG/').mkdir(parents=True, exist_ok=True)

#mask = params['masks']['dir_mask']+params['masks']['Tmaska']
#m    = hp.read_map(mask)


if idx==0: comp ='totfg'
if idx==1: comp ='rad'
if idx==2: comp ='cib'
if idx==3: comp ='tsz'
if idx==4: comp ='ksz'
if idx==5: comp ='cibtsz'


comps     = {
            'totfg' : {'name':'lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG', 'pol': True},
            'rad'   : {'name':'lradNG', 'pol': True},
            'cib'   : {'name':'lcibNG', 'pol': False},
            'tsz'   : {'name':'ltszNGbahamas80', 'pol': False},
            'ksz'   : {'name':'lkszNGbahamas80', 'pol': False},
            'cibtsz': {'name':'lcibNG_ltszNGbahamas80', 'pol': False},
            }
        

if comp=='none': fid1 = 'tszG_kszG_cibG_radG'
if comp=='tsz' : fid1 = 'tszNG_kszG_cibG_radG'
if comp=='ksz' : fid1 = 'tszG_kszNG_cibG_radG'
if comp=='cib' : fid1 = 'tszG_kszG_cibNG_radG'
if comp=='rad' : fid1 = 'tszG_kszG_cibG_radNG'
if comp=='all' : fid1 = 'tszNG_kszNG_cibNG_radNG'



cls_totfg = np.loadtxt('./fgcls/cls_totfg_%s.dat'%ilctype)[:5101,1:5]


#cmb = hp.read_map('/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.7/cmb/len/mdpl2_lensedTQU1_mdpl2phiNG_seed%d_lmax16000_nside8192_interp1.6_method1_pol_1_lensedmap.fits'%(seed+10000),field=[0,1,2])

# Adding Gaussian component
if comp == 'none':
    cls_totfg = np.loadtxt('./fgcls/cls_totfg_%s.dat'%ilctype)#[:5101,1:5]
    cls_totfg[:,1:5]=0 
    np.savetxt('./fgcls/cls_%s_%s_%s.dat'%(comp,suff,ilctype),cls_totfg)

if comp == 'all':
    cls_totfg = np.loadtxt('./fgcls/cls_totfg_%s.dat'%ilctype)#[:5101,1:5]
    np.savetxt('./fgcls/cls_%s_%s_%s.dat'%(comp,suff,ilctype),cls_totfg)

else:
    # Adding nonGaussian foregrounds
    fname   = dir_tmp+'/ilc/mdpl2_cmbs4w_%s_%s_uk_inpainted.fits'%(ilctype,comps[comp]['name'])

    if comp == 'ksz' or comp=='tsz' or comp=='cib':
        a = hp.read_map(fname,field=[0,1,2])
        a[1]=np.random.normal(loc=0.0, scale=1.0, size=a[0].shape[0])
        a[2]=np.random.normal(loc=0.0, scale=1.0, size=a[0].shape[0])
        fname = '/lcrc/project/SPT3G/users/ac.yomori/scratch/tmp%s.fits'%comp
        hp.write_map(fname,a,dtype=np.float32,overwrite=True)


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
                              clfile      = './fgcls/cls_%s_%s_%s.dat'%(comp,suff,ilctype),
                              nlmax=6500, apodizesigma=170, thetamax=180, beam=0, beam2=0, subav='YES', polarization='YES', decouple='YES'
                       )


