import sys
import yaml
import argparse
import pathlib
import healpy as hp
import numpy  as np

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml' , default=None, type=str, help='yaml file with all the configurations')
parser.add_argument('ilctype'   , default=None, type=str, help='cmbmv/ynull')
parser.add_argument('compidx'   , default=None, type=int, help='compidx 0-6')
args = parser.parse_args()

compidx = args.compidx
ilctype = args.ilctype

params  = yaml.safe_load(open(args.file_yaml))
dir_tmp = params['dirs']['dir_tmp']
expt    = params['expt']['expts'][0]

pathlib.Path(dir_tmp+'/ilc/').mkdir(parents=True, exist_ok=True)



#dir_base  = '/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.7/outputs/spt3g/bahamas80_scal1.000_randradioflux/'
   
comps     = {'totn' : {'name':'lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG', 'pol': True},
             'tot'  : {'name':'lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG', 'pol': True},
             'totfg': {'name':'lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG', 'pol': True},
             'cmb'  : {'name':'lcmbNG', 'pol': True},
             'rad'  : {'name':'lradNG', 'pol': True},
             'cib'  : {'name':'lcibNG', 'pol': True},
             'tsz'  : {'name':'ltszNGbahamas80', 'pol': True},
             'ksz'  : {'name':'lkszNGbahamas80', 'pol': True},
            }

dir_weights  = './weights/'
file_weightT = dir_weights+'weights_TT_'+params['runtime']['suff']+'_%s.dat'%ilctype
file_weightE = dir_weights+'weights_EE_'+params['runtime']['suff']+'_%s.dat'%ilctype
file_weightB = dir_weights+'weights_BB_'+params['runtime']['suff']+'_%s.dat'%ilctype

wT = np.loadtxt(file_weightT)
wE = np.loadtxt(file_weightE)
wB = np.loadtxt(file_weightB)

comp = list(comps)[compidx]

print("Processing %s"%comp)
file_map95  = dir_tmp+'/inpainted/mdpl2_%s_95ghz_%s_uk_inpainted.fits'%(expt,comps[comp]['name'])
file_map150 = dir_tmp+'/inpainted/mdpl2_%s_150ghz_%s_uk_inpainted.fits'%(expt,comps[comp]['name'])
file_map220 = dir_tmp+'/inpainted/mdpl2_%s_220ghz_%s_uk_inpainted.fits'%(expt,comps[comp]['name'])
#file_map285 = dir_tmp+'/inpainted/mdpl2_%s_285ghz_%s_uk_inpainted.fits'%(expt,comps[comp]['name'])


m95  = hp.read_map(file_map95,field=[0,1,2])
m150 = hp.read_map(file_map150,field=[0,1,2])
m220 = hp.read_map(file_map220,field=[0,1,2])

if comp=='totn':
    # add noise
    np.random.seed(0)
    nltt   = np.load('./noise/cls_noise_TT.npz')
    nlee   = np.load('./noise/cls_noise_EE.npz')
    nlbb   = np.load('./noise/cls_noise_BB.npz')

    nlm95   = hp.synalm([nltt['nl95'] ,nlee['nl95'] ,nlbb['nl95'] ,0*nltt['nl95']]  ,new=True)
    nlm150  = hp.synalm([nltt['nl150'],nlee['nl150'],nlbb['nl150'],0*nltt['nl150']] ,new=True)
    nlm220  = hp.synalm([nltt['nl220'],nlee['nl220'],nlbb['nl220'],0*nltt['nl220']] ,new=True)
    #nlm285  = hp.synalm([nltt['nl285'],nlee['nl285'],nlbb['nl285'],0*nltt['nl285']] ,new=True)

    nmap95  = hp.alm2map(nlm95 ,8192); m95 +=nmap95  ; del nmap95
    nmap150 = hp.alm2map(nlm150,8192); m150+=nmap150; del nmap150
    nmap220 = hp.alm2map(nlm220,8192); m220+=nmap220; del nmap220
    #nmap285 = hp.alm2map(nlm285,8192); m285+=nmap285; del nmap285

    
# Make all T Q U maps
alm95   = hp.map2alm(m95  ,lmax=6500, use_pixel_weights=True)
alm150  = hp.map2alm(m150 ,lmax=6500, use_pixel_weights=True)
alm220  = hp.map2alm(m220 ,lmax=6500, use_pixel_weights=True)
#alm285  = hp.map2alm(m285 ,lmax=6500, use_pixel_weights=True)

almmvT  = hp.almxfl(alm95[0],wT[0,:6501])+hp.almxfl(alm150[0],wT[1,:6501])+hp.almxfl(alm220[0],wT[2,:6501])
almmvE  = hp.almxfl(alm95[1],wE[0,:6501])+hp.almxfl(alm150[1],wE[1,:6501])+hp.almxfl(alm220[1],wE[2,:6501])
almmvB  = hp.almxfl(alm95[2],wB[0,:6501])+hp.almxfl(alm150[2],wB[1,:6501])+hp.almxfl(alm220[2],wB[2,:6501])

dmap    = hp.alm2map([almmvT,almmvE,almmvB],8192)

if comp=='totn': comps[comp]['name']+='_noise'

hp.write_map(dir_tmp+'/ilc/mdpl2_%s_%s_%s_uk_inpainted.fits'%(expt,ilctype,comps[comp]['name']), dmap, overwrite=True, dtype=np.float32)
