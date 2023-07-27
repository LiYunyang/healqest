""" 
Given the total powerspectrum in 3 SPT channels, use 
Srini's code to compute the frequency weights. This
code needs 3g_software to run.

On sherlock:
source  /home/groups/kipac/yomori/packages/3g_helper/setup/setup_sherlock.sh
python3 get_freqweights.py configs/configs/config_spt3g_fulldepth_paper.yaml 

On crossover:
python3 get_freqweights_theorycmb.py configs/config_spt3g_fulldepth_paper.yaml T --smoothcls

***Note for the future: this code generates weights for T/E/B separately whereas Srini's
original code has the capability to generate the joint weights***
"""

import os
import sys
sys.path.insert(0,'/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_curvlens/spt3g_software_curvlens/build/')
import camb
import datetime
import argparse
import numpy as np
import yaml
import healpy as hp
import subprocess
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software/ilc/python')
import ilc
from pathlib import Path
from scipy.signal import savgol_filter as sf

def smoothcls(cls):
    l=np.arange(len(cls[:,0]))
    for i in range(0,cls.shape[1]):
        tmp=sf(l**2*cls[:,i],151,3)/l**2
        tmp[0]=0
        cls[:,i]=tmp
    return cls

np.seterr(all='ignore')

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml'   , default=None, type=str, help='yaml file with all the configurations')
parser.add_argument('mode'         , default=None, type=str, help='mode (T, E or B)')
parser.add_argument('--get_fgcls' , default=False, dest='get_fgcls' ,action='store_true')
args = parser.parse_args()

params  = yaml.safe_load(open(args.file_yaml))

expt        = params['expt']['expts']
freqs       = params['expt']['freqs']
dir_mdpl    = params['dirs']['dir_mdpl']#+'/outputs/%s/bahamas80_scal1.000_randradioflux/'%expt
dir_tmp     = params['dirs']['dir_tmp']
#dir_out     = dir_tmp+'/'+'/weights/'
dir_weights =  params['dirs']['dir_weights']

if   args.mode=='T': which_spec='TT' 
elif args.mode=='E': which_spec='EE'
elif args.mode=='B': which_spec='BB'
else: sys.exit('undefined mode, must be T/E/B')

# Load config
params  = yaml.safe_load(open(args.file_yaml))

# Check lmax 
lmax    = params['runtime']['lmax']
ell     = np.arange(lmax+1)

# Extract the frequency channel names
freqs   = params['expt']['freqs']
expts   = params['expt']['expts']

# Store information about each channel in a dictionary
freq_info = {}
for i in range(0,len(freqs)):
    freq_info[freqs[i]] = [ 
                            params['expt']['fwhm'][i], 
                            params['expt']['nlevt'][i],
                            params['expt']['lkneet'][i], 
                            params['expt']['akneet'][i],
                            params['expt']['nlevp'][i],
                            params['expt']['lkneep'][i], 
                            params['expt']['akneep'][i]
                          ] 


nl1d_dic = {}

print("Computing noise spectra for %s"%which_spec)

nl1d_dic[which_spec] = {}

for freq in freqs:
    fwhm, nlevt, lkneet, akneet, nlevp, lkneep, akneep  =  freq_info[freq]
            
    if which_spec == 'TT':
        lknee  = lkneet
        aknee  = akneet
        nlev   = nlevt
    elif which_spec == 'EE' or which_spec == 'BB':
        lknee  = lkneep
        aknee  = akneep
        nlev   = nlevp
    elif which_spec == 'TE':
        lknee  = lkneep
        nlev   = 0
            
    nl1d = (np.pi/180./60.*nlev)**2 / hp.gauss_beam(fwhm=np.radians(fwhm/60.),lmax=lmax)**2 * (1.+(lknee*1./ell)**aknee )
    nl1d[0]=0

    nl1d_dic[which_spec][freq] = nl1d

Path('./noise/').mkdir(parents=True, exist_ok=True)
np.savez('.//noise/cls_noise_%s.npz'%which_spec,nl95  = nl1d_dic[which_spec][95]
                                               ,nl150 = nl1d_dic[which_spec][150]
                                               ,nl220 = nl1d_dic[which_spec][220]
                                               #,nl285 = nl1d_dic[which_spec][285]
                                                )


# Setting up the component separation
comp_dict_for_ilc = {}
comp_dict_for_ilc['cmbmv']    = ['CMB', None , ['CMB', 'kSZ']]
comp_dict_for_ilc['cmbynull'] = ['CMB', ['Y'], ['CMB', 'kSZ']]
#comp_dict_for_ilc['cibnull']  = ['CMB', [ params['ilc']['cibnull'][0], params['ilc']['cibnull'][1] ], ['CMB', 'kSZ']] 

# Generate theory cls
cls_dict             = {}
cls_dict[which_spec] = {}

for i in range(0,len(freqs)):
    for j in range(0,len(freqs)):
        expt1 = expts[i]
        expt2 = expts[j]
        freq1 = freqs[i]
        freq2 = freqs[j]

        if which_spec=='TT':idx=1
        if which_spec=='EE':idx=2
        if which_spec=='BB':idx=3
        
        cls_fg = np.loadtxt('./fgcls/cls_totfg_%s_%dghz_%s_%dghz.dat'%(expt1,freq1,expt2,freq2) )[:,idx]
    
        # Add noise if auto
        if i==j:
            nl=nl1d_dic[which_spec][freqs[i]]
        else:
            nl=0

        cls_dict[which_spec][('%sGHz'%freqs[i], '%sGHz'%freqs[j])] = cls_fg + nl


freqnames=[]
for i in range(0,len(freqs)):
    freqnames.append('%dGHz'%freqs[i])

# Run the ilc code to compute weights
ilc_dict = {}
ilc_dict[which_spec] = {}

print("performing ILC for %s"%which_spec)  

for keyname in comp_dict_for_ilc:
    final_comp, null_comp, ignore_fg = comp_dict_for_ilc[keyname]
    print(keyname, final_comp, null_comp, ignore_fg)
    curr_weights_arr, curr_cl_residual_arr = ilc.perform_ilc( 
                                                             final_comp,
                                                             freqnames , 
                                                             expts     , 
                                                             lmax      ,
                                                             lmin    = 10 ,
                                                             spec    = which_spec, 
                                                             cl_dict = cls_dict, 
                                                             nl_dict = nl1d_dic,
                                                             ell     = ell,
                                                             null_components = null_comp,
                                                             ignore_fg       = ignore_fg
                                                            )  

    ilc_dict[which_spec][keyname] = [curr_weights_arr[:,0], curr_cl_residual_arr[0]]

    #Save weight file
    Path('./weights/').mkdir(parents=True, exist_ok=True)
    file_out = './weights/weights_%s_'%which_spec+params['runtime']['suff']+'_%s.dat'%keyname
    np.savetxt(file_out ,curr_weights_arr[:,0])
