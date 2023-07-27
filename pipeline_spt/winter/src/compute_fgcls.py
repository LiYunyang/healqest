import sys
import yaml
import pathlib
import argparse
import healpy as hp
import numpy as np
import sys,os
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/skymapkit/src/utils/')
import utils
import pathlib

os.environ['HEALPIX'] = "/lcrc/project/SPT3G/users/ac.yomori/envs/analysis/Healpix_3.80/"
spice='/lcrc/project/SPT3G/users/ac.yomori/packages/PolSpice_v03-07-03/bin/spice'


parser = argparse.ArgumentParser()
parser.add_argument('file_yaml'  , default=None, type=str, help='yaml file with all the configurations')
parser.add_argument('freqidx'    , default=0   , type=int, help='param index')
parser.add_argument('comp'       , default=0   , type=str, help='comp')
args = parser.parse_args()

params  = yaml.safe_load(open(args.file_yaml))
freqidx = args.freqidx
comp    = args.comp

expts    = params['expt']['expts']
freqs    = params['expt']['freqs']
dir_mdpl = params['dirs']['dir_mdpl']
dir_tmp  = params['dirs']['dir_tmp']
dir_out  = './fgcls/'


pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)


comps     = {'tot'  : {'name':'lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG', 'pol': True},
             'totfg': {'name':'lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG', 'pol': True},
             'cmb'  : {'name':'lcmbNG', 'pol': True},
             'rad'  : {'name':'lradNG', 'pol': True},
             'cib'  : {'name':'lcibNG', 'pol': False},
             'tsz'  : {'name':'ltszNGbahamas80', 'pol': False},
             'ksz'  : {'name':'lkszNGbahamas80', 'pol': False},
            }


freqcombs=[]
c=0
for c1 in range(0,len(freqs)):
    for c2 in range(0,len(freqs)):
        expt1 = expts[c1]
        freq1 = freqs[c1]
        expt2 = expts[c2]
        freq2 = freqs[c2]
        print(expt1,freq1,expt2,freq2)
        freqcombs.append((expt1,freq1,expt2,freq2))

expt1 = freqcombs[freqidx][0]
freq1 = freqcombs[freqidx][1]
expt2 = freqcombs[freqidx][2]
freq2 = freqcombs[freqidx][3]
        
file_map1  = dir_tmp + '/inpainted/mdpl2_%s_%dghz_%s_uk_inpainted.fits'%(expt1,freq1,comps[args.comp]['name'])
file_map2  = dir_tmp + '/inpainted/mdpl2_%s_%dghz_%s_uk_inpainted.fits'%(expt2,freq2,comps[args.comp]['name'])

print("- Reading masks")
file_maskT = params['masks']['dir_mask']+params['masks']['Tmaska']
file_maskP = params['masks']['dir_mask']+params['masks']['Pmask']
maskT      = hp.read_map(file_maskT)
maskP      = hp.read_map(file_maskP)

if comps[comp]['pol']==True: pol='YES'; dec='YES'
else:  pol='NO'; dec='NO'
utils.run_polspice(spice, mapfile     = file_map1,
                          mapfile2    = file_map2,
                          weightfile  = file_maskT,
                          weightfile2 = file_maskT,
                          clfile      = dir_out+'/cls_%s_%s_%dghz_%s_%dghz.dat'%(comp,expt1,freq1,expt2,freq2),
                          nlmax=6500, apodizesigma=170, thetamax=180, beam=0, beam2=0, subav='YES', polarization=pol, decouple=dec
                   )

