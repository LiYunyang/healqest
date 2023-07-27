import sys
import yaml
import argparse
import pathlib
import healpy as hp
import numpy  as np
import os
import subprocess

os.environ['HEALPIX'] = "/lcrc/project/SPT3G/users/ac.yomori/envs/analysis/Healpix_3.80/"
spice='/lcrc/project/SPT3G/users/ac.yomori/packages/PolSpice_v03-07-03/bin/spice'


parser = argparse.ArgumentParser()
parser.add_argument('file_yaml' , default=None, type=str, help='yaml file with all the configurations')
parser.add_argument('ilctype'   , default=None, type=str, help='cmbmv/ynull')
args = parser.parse_args()

ilctype = args.ilctype

params  = yaml.safe_load(open(args.file_yaml))
dir_tmp = params['dirs']['dir_tmp']
suff    = params['runtime']['suff'] 

pathlib.Path(dir_tmp+'/filter/').mkdir(parents=True, exist_ok=True)


ret = np.zeros((6501,7)) # to match the dimension of alm2cl


#-------------   noise   -----------------
for c,spec in enumerate(['TT','EE','BB']):
    print("Making noise specrta for %s"%spec)
    w    = np.loadtxt('./weights/weights_%s_%s_%s.dat'%(spec,suff,ilctype))
    nl   = np.load('./noise/cls_noise_%s.npz'%spec) 

    for i in range(0,6501):
        clmat= np.zeros((3,3))
        clmat[0,0] = nl['nl95'][i]
        clmat[1,1] = nl['nl150'][i]
        clmat[2,2] = nl['nl220'][i]
        #clmat[3,3] = nl['nl285'][i]

        ret[i,c+1]=np.dot(w[:,i], np.dot(clmat, w[:,i].T))


ret[:,0]=np.arange(6501)
l=np.arange(6501)
np.savetxt('./noise/cls_noise_%s.dat'%ilctype,ret)

# ------------- foreground ---------------
mask = params['masks']['dir_mask']+params['masks']['Tmaska']

print(mask)
print( dir_tmp+'/ilc/mdpl2_%s_%s_lcibNG_ltszNGbahamas80scaled1.000_lkszNGbahamas80_lradNG_uk_inpainted.fits'%(suff,ilctype) )
print('./fgcls/cls_totfg_%s.dat'%ilctype)
subprocess.call([spice,    '-mapfile'     , dir_tmp+'/ilc/mdpl2_%s_%s_lcibNG_ltszNGbahamas80scaled1.000_lkszNGbahamas80_lradNG_uk_inpainted.fits'%(suff,ilctype),
                           '-weightfile'  , mask,
                           '-clfile'      , './fgcls/cls_totfg_%s.dat'%ilctype,
                           '-nlmax'       ,'7000',
                           '-apodizesigma','170',
                           '-thetamax'    ,'180',
                           '-subav'       ,'YES',
                           '-verbosity'   , 'YES',
                           '-polarization', 'YES',
                           '-decouple'    , 'YES'
                        ])


#-------------  noise+fg   -----------------
for c,spec in enumerate(['TT','EE','BB']):
    print("Making noise specrta for %s"%spec)
    w    = np.loadtxt('./weights/weights_%s_%s_%s.dat'%(spec,suff,ilctype))
    nl   = np.load('./noise/cls_noise_%s.npz'%spec)

    if spec=='TT': idx=1
    if spec=='EE': idx=2
    if spec=='BB': idx=3

    fg_95_95   = np.loadtxt('./fgcls/cls_totfg_%s_95ghz_%s_95ghz.dat'%(suff,suff) )[:,idx]
    fg_95_150  = np.loadtxt('./fgcls/cls_totfg_%s_95ghz_%s_150ghz.dat'%(suff,suff) )[:,idx]
    fg_95_220  = np.loadtxt('./fgcls/cls_totfg_%s_95ghz_%s_220ghz.dat'%(suff,suff) )[:,idx]
    #fg_95_285  = np.loadtxt('./fgcls/cls_totfg_cmbs4w_95ghz_cmbs4w_285ghz.dat')[:,idx]
    fg_150_150 = np.loadtxt('./fgcls/cls_totfg_%s_150ghz_%s_150ghz.dat'%(suff,suff) )[:,idx]
    fg_150_220 = np.loadtxt('./fgcls/cls_totfg_%s_150ghz_%s_220ghz.dat'%(suff,suff) )[:,idx]
    #fg_150_285 = np.loadtxt('./fgcls/cls_totfg_cmbs4w_150ghz_cmbs4w_285ghz.dat')[:,idx]
    fg_220_220 = np.loadtxt('./fgcls/cls_totfg_%s_220ghz_%s_220ghz.dat'%(suff,suff))[:,idx]
    #fg_220_285 = np.loadtxt('./fgcls/cls_totfg_cmbs4w_220ghz_cmbs4w_285ghz.dat')[:,idx]
    #fg_285_285 = np.loadtxt('./fgcls/cls_totfg_cmbs4w_285ghz_cmbs4w_285ghz.dat')[:,idx]

    for i in range(0,6501):
        clmat= np.zeros((3,3))
        clmat[0,0] = nl['nl95'][i]  + fg_95_95[i]
        clmat[1,1] = nl['nl150'][i] + fg_150_150[i]
        clmat[2,2] = nl['nl220'][i] + fg_220_220[i]
        #clmat[3,3] = nl['nl285'][i] + fg_285_285[i]

        clmat[0,1] = clmat[1,0] = fg_95_150[i]
        clmat[0,2] = clmat[2,0] = fg_95_220[i]
        #clmat[0,3] = clmat[3,0] = fg_95_285[i]

        clmat[1,2] = clmat[2,1] = fg_150_220[i]
        #clmat[1,3] = clmat[3,1] = fg_150_285[i]

        #clmat[2,3] = clmat[3,2] = fg_220_285[i]

        ret[i,c+1]=np.dot(w[:,i], np.dot(clmat, w[:,i].T))


ret[:,0]=np.arange(6501)
l=np.arange(6501)
np.savetxt('./noise/cls_noise_totfg_%s.dat'%ilctype,ret)

