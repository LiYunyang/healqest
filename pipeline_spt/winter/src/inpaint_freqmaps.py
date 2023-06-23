import sys
import yaml
import pathlib
import argparse
import healpy as hp
import numpy as np

def itersmooth(inmap,mask,R):#,dec,R):
    invmask = mask*-1 + 1
    a       = np.copy(inmap) * mask
    for Ri in (R):
        print(Ri)
        a     = hp.smoothing(a,fwhm=Ri*0.000290888,use_pixel_weights=True,lmax=8192)
        a     = inmap*mask+a*invmask
    return a

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml'  , default=None, type=str, help='yaml file with all the configurations')
parser.add_argument('compidx'    , default=0   , type=int, help='param index')
args = parser.parse_args()

params  = yaml.safe_load(open(args.file_yaml))
compidx = args.compidx

expts    = params['expt']['expts']
freqs    = params['expt']['freqs']
dir_mdpl = params['dirs']['dir_mdpl']
dir_tmp  = params['dirs']['dir_tmp']
dir_out  = dir_tmp+'/inpainted/'

pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

comps      = [
             'lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG',
             'lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG',
             'lcmbNG',
             'lradNG',
             'lcibNG',
             'ltszNGbahamas80',
             'lkszNGbahamas80',
             ]


print("- Reading masks")
file_maskT = params['masks']['dir_mask']+params['masks']['Tmask']
file_maskP = params['masks']['dir_mask']+params['masks']['Pmask']
maskT      = hp.read_map(file_maskT)
maskP      = hp.read_map(file_maskP)

rnQ = np.random.normal(size=hp.nside2npix(8192))*1e-30
rnU = np.random.normal(size=hp.nside2npix(8192))*1e-30

print("- Inpainting and save all frequencies")
for c,freq in enumerate(freqs):
    print(freq)
    
    if compidx >3:
        # Load only T maps
        R_T   = np.arange(6,0,-2)
        T     = hp.read_map(dir_mdpl+'mdpl2_%s_%dghz_%s_uk.fits'%(expts[c],freq,comps[compidx]),field=0)
        Ts    = itersmooth(T, maskT, R=R_T)
        zero  = np.zeros_like(Ts)
        hp.write_map(dir_out+'/mdpl2_%s_%dghz_%s_uk_inpainted.fits'%(expts[c],freq,comps[compidx]), [Ts,rnQ,rnU], dtype=np.float32,overwrite=True)

    else:
        # Load T/Q/U maps
        R_T   = np.arange(6,0,-2)
        R_P   = np.ones(3)
        T,Q,U = hp.read_map(dir_mdpl+'mdpl2_%s_%dghz_%s_uk.fits'%(expts[c],freq,comps[compidx]),field=[0,1,2])
        Ts    = itersmooth(T, maskT, R=R_T )
        Qs    = itersmooth(Q, maskP, R=R_P )
        Us    = itersmooth(U, maskP, R=R_P )
        hp.write_map(dir_out+'/mdpl2_%s_%dghz_%s_uk_inpainted.fits'%(expts[c],freq,comps[compidx]), [Ts,Qs,Us],dtype=np.float32,overwrite=True )
