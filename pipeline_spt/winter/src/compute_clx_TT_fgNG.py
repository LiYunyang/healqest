import os
import sys
import healpy as hp
import numpy as np
import subprocess
import yaml
import argparse 
import pathlib 

os.environ['HEALPIX'] = "/lcrc/project/SPT3G/users/ac.yomori/envs/analysis/Healpix_3.80/"
spice='/lcrc/project/SPT3G/users/ac.yomori/packages/PolSpice_v03-07-03/bin/spice'

#parser = argparse.ArgumentParser()
#parser.add_argument('file_yaml' , default=None, type=str, help='yaml file with all the configurations')
#args = parser.parse_args()

#dir_mask = '/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.6/outputs/spt3g/bahamas80_scal1.000/ilc/masks/'
#mask     = dir_mask+'mask8192_mdpl2_lenmag_cibmap_radmap_fluxgt6mjy_150ghz_f25mjy_f500mjy_r1_r1_r1_apod3_clusters_mgt2e14_apod10.fits'
#mask     = '/lcrc/globalscratch/ac.yomori/mdpl2_lensingbiases/masks/mask8192_mdpl2_lenmag_cibmap_radmap_spt3g_150ghz_fluxcut6.0mjy_singlepix_clusters_spt3g_7sigma_apod5.fits'


file_yaml = str(sys.argv[1])
ilctype   = str(sys.argv[2])
i         = int(sys.argv[3])
comp1     = str(sys.argv[4])
comp2     = str(sys.argv[5])
fgseed    = int(sys.argv[6])

params  = yaml.safe_load(open(file_yaml))
mask = params['masks']['dir_mask']+'mask8192_mdpl2_clusters_cmbs4w_10sigma_apod3.fits'

l=np.arange(5101)

dir_p    = '/lcrc/globalscratch/ac.yomori/crossilc/spt3g/lensrec_%s/'%ilctype
#dir_p2   = '/lcrc/globalscratch/ac.yomori/crossilc/cmbs4w/lensrec_mfresp/'
dir_out  = dir_p#'/lcrc/globalscratch/ac.yomori/crossilc/cmbs4w/clkk_fgNG/'

pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

respavgTT = np.load(dir_p+'/respavgTT.npz')['resp']


if i==0:
    qid= 'ktt'
    w  = respavgTT
    w[-100:]=np.inf


    if comp1 == 'input':
        print('Using input kappa')
        file_k = '/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.7/cmbkappa/raytrace16384_ip20_cmbkappa.zs1.kg1g2_highzadded_lowLcorrected.fits'
   
    else:
        print('Using reconstructed kappa')
        print(dir_p+'/plmTT_%da_%da_%s_%s.alm.npz'%(i,i,comp1,comp1))
        ylmTTn0=np.load(dir_p+'/plmTT_%da_%da_%s_%s.alm.npz'%(i,i,comp1,comp1))['glm']
        klmk  = hp.almxfl(ylmTTn0,0.5*l*(l+1)*1/w)
        kmapk = hp.alm2map(klmk,8192)
        hp.write_map('/tmp/kmapk1_%s_%d.fits'%(comp1,i),kmapk,overwrite=True,dtype=np.float32)
        file_k='/tmp/kmapk1_%s_%d.fits'%(comp1,i)


    for z in range(1,6):
        print(z)
        if comp2 == 'dens':
            gmap = '/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.5/lss/density/mdpl2_biaseddensity_finesamplednz_len_lsst_y1_lens_zbin%d_fullsky.fits'%z
            xcor='nk'

        if comp2 == 'shear':
            gmap = '/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.5/lss/shear/raytrace16384_ip20.kg1g2_nside16384_lsst_y1_srcs_zbin%d_fullsky_lmax24576.fits'%z
            xcor='gk'

        subprocess.call([spice,'-mapfile'     , file_k,
                               '-weightfile'  , mask,
                               '-mapfile2'    , gmap,
                               '-weightfile2' , mask,
                               '-clfile'      , dir_out+'cl%s_%s_%da_%da_%da_%da_%s_%s_zbin%d.dat'%(xcor,qid,i,i,i,i,comp1,comp2,z),
                               '-nlmax'       ,'5100',
                               '-apodizesigma','170',
                               '-thetamax'    ,'180',
                               '-subav'       ,'YES',
                               '-verbosity'   , 'NO',
                            ])
    
    if comp1 != 'input':
        os.remove( file_k)


