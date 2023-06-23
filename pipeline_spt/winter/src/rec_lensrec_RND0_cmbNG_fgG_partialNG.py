'''
python rec.py 'TT' 100 4000 6000 mdpl2phiNG 0 1 0 1 rad cib
'''
import sys
import healpy as hp
import numpy as np
import camb
import yaml
import argparse
sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/healqest/healqest/src/')
import utils
import weights
import qest
import wignerd,resp
from pathlib import Path
#sys.path.append('/lcrc/project/SPT3G/users/ac.yomori/repo/Quicklens-with-fixes/')
#import quicklens as ql


def reduce_lmax(alm, lmax=4000):
        lmaxin  = hp.Alm.getlmax(alm.shape[0])
        print( "reducing lmax: lmax_in=%g -> lmax_out=%g"%(lmaxin,lmax) )
        ell,emm = hp.Alm.getlm(lmaxin)
        almout  = np.zeros(hp.Alm.getsize(lmax),dtype=np.complex_)
        oldi=0
        oldf=0
        newi=0
        newf=0
        dl = lmaxin-lmax
        for i in range(0,lmax+1):
                oldf=oldi+lmaxin+1-i
                newf=newi+lmax+1-i
                almout[newi:newf]=alm[oldi:oldf-dl]
                oldi=oldf
                newi=newf
        return almout


parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='dir_base')
#parser.add_argument('ilctype'  , default=None, type=str, help='ilctype')
parser.add_argument('qetype'   , default=None, type=str, help='yaml file with all the configurations')
parser.add_argument('seed1'    , default=1   , type=int, help='seed1')
parser.add_argument('cmbset1'  , default=1   , type=int, help='cmbset1')
parser.add_argument('seed2'    , default=1   , type=int, help='seed2')
parser.add_argument('cmbset2'  , default=1   , type=int, help='cmbset2')
parser.add_argument('comp1'    , default=1   , type=str, help='comp')
parser.add_argument('comp2'    , default=1   , type=str, help='comp')
parser.add_argument('fgseed'   , default=1   , type=int, help='fgseed')
args = parser.parse_args()

#  = args.yaml
#ilctype   = args.ilctype
seed1     = args.seed1
cmbset1   = args.cmbset1
seed2     = args.seed2
cmbset2   = args.cmbset2
comp1     = args.comp1
comp2     = args.comp2
fgseed    = args.fgseed

params    = yaml.safe_load(open(args.file_yaml))
dir_tmp   = params['dirs']['dir_tmp']
suff      = params['runtime']['suff']
dir_mask  = params['masks']['dir_mask']
Tmaska    = params['masks']['Tmaska']

lmin      = params['lensing']['lmin']
lmaxT     = params['lensing']['lmaxT']
lmaxP     = params['lensing']['lmaxP']
Lmax      = params['lensing']['Lmax']

lmaxTP = max(lmaxT,lmaxP)

if comp1=='none': fid1 = 'phiNG_tszG_kszG_cibG_radG'
if comp1=='tsz' : fid1 = 'phiNG_tszNG_kszG_cibG_radG'
if comp1=='ksz' : fid1 = 'phiNG_tszG_kszNG_cibG_radG'
if comp1=='cib' : fid1 = 'phiNG_tszG_kszG_cibNG_radG'
if comp1=='rad' : fid1 = 'phiNG_tszG_kszG_cibG_radNG'
if comp1=='all' : fid1 = 'phiNG_tszNG_kszNG_cibNG_radNG'

if comp2=='none': fid2 = 'phiNG_tszG_kszG_cibG_radG'
if comp2=='tsz' : fid2 = 'phiNG_tszNG_kszG_cibG_radG'
if comp2=='ksz' : fid2 = 'phiNG_tszG_kszNG_cibG_radG'
if comp2=='cib' : fid2 = 'phiNG_tszG_kszG_cibNG_radG'
if comp2=='rad' : fid2 = 'phiNG_tszG_kszG_cibG_radNG'
if comp2=='all' : fid2 = 'phiNG_tszNG_kszNG_cibNG_radNG'

clini     = '/lcrc/project/SPT3G/users/ac.yomori/repo/skymapkit/data/camb//mdpl2_params.ini'
clfile    = '/lcrc/project/SPT3G/users/ac.yomori/repo/skymapkit/data/camb/mdpl2_lensedCls.dat'
mask      = hp.read_map(dir_mask+Tmaska)
pars      = camb.read_ini(clini)
results   = camb.get_results(pars)
sltt,slee,slbb,slte  = results.get_cmb_power_spectra(pars,lmax=lmaxTP, CMB_unit='muK',raw_cl=True)['lensed_scalar'].T

#-------- Loading map 1 ---------
if seed1==0:
    ilctype='cmbynull'
    tlm1,elm1,blm1 = hp.read_alm(dir_tmp+'/gaussmaps_cmbNG_fgG_partialNG/cmb%d_%s_%s_0_gauss%d.alm'%(cmbset1,ilctype,fid1,fgseed),hdu=[1,2,3])
else:
    #Fully Gaussain foregeound
    ilctype='cmbynull'
    tlm1,elm1,blm1 = hp.read_alm(dir_tmp+'/gaussmaps/cmb%d_%s_%s_%d.alm'%(cmbset1,ilctype,'phiG_tszG_kszG_cibG_radG',seed1),hdu=[1,2,3])
    fid1 = 'phiG_tszG_kszG_cibG_radG'

if seed2==0:
    ilctype='cibnull'
    tlm2,elm2,blm2 = hp.read_alm(dir_tmp+'/gaussmaps_cmbNG_fgG_partialNG/cmb%d_%s_%s_0_gauss%d.alm'%(cmbset2,ilctype,fid2,fgseed),hdu=[1,2,3])
else:
    # Fully Gaussian foreground
    ilctype='cibnull'
    tlm2,elm2,blm2 = hp.read_alm(dir_tmp+'/gaussmaps/cmb%d_%s_%s_%d.alm'%(cmbset2,ilctype,'phiG_tszG_kszG_cibG_radG',seed2),hdu=[1,2,3])
    fid2 = 'phiG_tszG_kszG_cibG_radG'

#cltotres =np.loadtxt(dir_tmp+'/filter/filter_cls_fg_noise_%s.dat'%ilctype) 
ilctype='cmbynull'
cls_noise = np.loadtxt('./noise/cls_noise_%s.dat'%ilctype)[:5101,1:5]
cls_totfg = np.loadtxt('./fgcls/cls_totfg_%s.dat'%ilctype)[:5101,1:5]
cltotres1  = cls_totfg + cls_noise

ilctype='cibnull'
cls_noise = np.loadtxt('./noise/cls_noise_%s.dat'%ilctype)[:5101,1:5]
cls_totfg = np.loadtxt('./fgcls/cls_totfg_%s.dat'%ilctype)[:5101,1:5]
cltotres2  = cls_totfg + cls_noise

# Construct filter1
flT1 = 1.0/(sltt+cltotres1[:lmaxTP+1,0]); flT1[lmaxT+1:] = 0; flT1[:lmin] = 0
flE1 = 1.0/(slee+cltotres1[:lmaxTP+1,1]); flE1[lmaxP+1:] = 0; flE1[:lmin] = 0
flB1 = 1.0/(slbb+cltotres1[:lmaxTP+1,2]); flB1[lmaxP+1:] = 0; flB1[:lmin] = 0

flT2 = 1.0/(sltt+cltotres2[:lmaxTP+1,0]); flT2[lmaxT+1:] = 0; flT2[:lmin] = 0
flE2 = 1.0/(slee+cltotres2[:lmaxTP+1,1]); flE2[lmaxP+1:] = 0; flE2[:lmin] = 0
flB2 = 1.0/(slbb+cltotres2[:lmaxTP+1,2]); flB2[lmaxP+1:] = 0; flB2[:lmin] = 0


if args.qetype[0]=='T': tlm1 = utils.reduce_lmax(tlm1,lmax=lmaxTP); almbar1 = hp.almxfl(tlm1,flT1); flm1= flT1
if args.qetype[0]=='E': elm1 = utils.reduce_lmax(elm1,lmax=lmaxTP); almbar1 = hp.almxfl(elm1,flE1); flm1= flE1
if args.qetype[0]=='B': blm1 = utils.reduce_lmax(blm1,lmax=lmaxTP); almbar1 = hp.almxfl(blm1,flB1); flm1= flB1

if args.qetype[1]=='T': tlm2 = utils.reduce_lmax(tlm2,lmax=lmaxTP); almbar2 = hp.almxfl(tlm2,flT2); flm2= flT2
if args.qetype[1]=='E': elm2 = utils.reduce_lmax(elm2,lmax=lmaxTP); almbar2 = hp.almxfl(elm2,flE2); flm2= flE2
if args.qetype[1]=='B': blm2 = utils.reduce_lmax(blm2,lmax=lmaxTP); almbar2 = hp.almxfl(blm2,flB2); flm2= flB2


if cmbset1==1: cmbset1name='a'
if cmbset1==2: cmbset1name='b'
if cmbset2==1: cmbset2name='a'
if cmbset2==2: cmbset2name='b'

print('Running lensing reconstruction')
glm,clm = qest.qest(args.qetype,Lmax,clfile,almbar1,almbar2) #reconstruct TT lensing

respP   = resp.fill_resp(weights.weights(args.qetype,Lmax,clfile), np.zeros(Lmax+1, dtype=np.complex_), flm1, flm2)


Path(dir_tmp+'/lensrec_cmbNG_fgG_partialNG/fgseed%d/%s/'%(fgseed,args.qetype)).mkdir(parents=True, exist_ok=True)
np.savez(dir_tmp+'/lensrec_cmbNG_fgG_partialNG/fgseed%d/%s/plm%s_%d%s_%d%s_%s_x_%s_fgseed%d.alm'%(fgseed,args.qetype,args.qetype,seed1,cmbset1name,seed2,cmbset2name,fid1,fid2,fgseed),glm=glm,analytical_resp=respP)
print("done")


#qest_TT = ql.qest.lens.phi_TT(sltt[:Lmax+1])
#resp = qest_TT.fill_resp(qest_TT, np.zeros(Lmax+1, dtype=np.complex), flm1,flm2)
#l=np.arange(Lmax+1)
#np.savetxt('/lcrc/project/SPT3G/users/ac.yomori/scratch/lensingtest5/resp_analytical.dat',np.c_[l,resp.real])
