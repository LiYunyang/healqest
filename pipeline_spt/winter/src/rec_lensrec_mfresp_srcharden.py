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

def srchard_weighting(ee, es, ss, weight):
    """
      ee, es, ss: response or N0 of est*est, est*src, src*src
      weight: - R^es/R^ss
      output: weighted response/N0
    """
    return ee + 2*weight*es + weight**2*ss

def harden_est(plmf1, plmf2, resp12):
    plm1  = plmf1['glm']
    plm2  = plmf2['glm']
    resp1 = plmf1['analytical_resp']
    resp2 = plmf2['analytical_resp']

    weight = -1*resp12 / resp2
    plm    = plm1 + hp.almxfl(plm2, weight)
    resp   = srchard_weighting(resp1,resp12,resp2,weight)
    return plm, resp

def resp_xest(qe1,qe2,clfile,cls,dict_lrange):
    #cross-estimator response
    assert qe1[:2] == qe2[:2], "input maps must be identical"
    Lmax = dict_lrange['Lmax']

    flT,flE,flB    = get_fl(cls, dict_lrange)
    if qe1[0]=='T': flm1=flT
    if qe1[1]=='T': flm2=flT
    if qe1[0]=='E': flm1=flE
    if qe1[1]=='E': flm2=flE
    if qe1[0]=='B': flm1=flB
    if qe1[1]=='B': flm2=flB
    aresp   = resp.fill_resp(weights.weights(qe1,Lmax,clfile),
                             np.zeros(Lmax+1, dtype=np.complex_), flm1, flm2,
                             qeZA=weights.weights(qe2,Lmax,clfile))
    return aresp


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
parser.add_argument('ilctype'  , default=None, type=str, help='ilctype')
parser.add_argument('qetype'   , default=None, type=str, help='yaml file with all the configurations')
parser.add_argument('seed1'    , default=1   , type=int, help='seed1')
parser.add_argument('cmbset1'  , default=1   , type=int, help='cmbset1')
parser.add_argument('seed2'    , default=1   , type=int, help='seed2')
parser.add_argument('cmbset2'  , default=1   , type=int, help='cmbset2')
parser.add_argument('comp1'    , default=1   , type=str, help='comp')
parser.add_argument('comp2'    , default=1   , type=str, help='comp')
args = parser.parse_args()

#  = args.yaml
ilctype   = args.ilctype
seed1     = args.seed1
cmbset1   = args.cmbset1
seed2     = args.seed2
cmbset2   = args.cmbset2
comp1     = args.comp1
comp2     = args.comp2

params    = yaml.safe_load(open(args.file_yaml))
dir_tmp   = params['dirs']['dir_tmp']
suff      = params['runtime']['suff']
dir_mask  = params['masks']['dir_mask']
Tmaska    = params['masks']['Tmaska']

lmin      = params['lensing']['lmin']
lmaxT     = params['lensing']['lmaxT']
lmaxP     = params['lensing']['lmaxP']
Lmax      = params['lensing']['Lmax']

lmaxTP    = max(lmaxT,lmaxP)

clini     = '/lcrc/project/SPT3G/users/ac.yomori/repo/skymapkit/data/camb//mdpl2_params.ini'
clfile    = '/lcrc/project/SPT3G/users/ac.yomori/repo/skymapkit/data/camb/mdpl2_lensedCls.dat'
mask      = hp.read_map(dir_mask+Tmaska)
pars      = camb.read_ini(clini)
results   = camb.get_results(pars)
sltt,slee,slbb,slte  = results.get_cmb_power_spectra(pars,lmax=lmaxTP, CMB_unit='muK',raw_cl=True)['lensed_scalar'].T

if seed1==0:
    cid1 = 'phiNG'
    sys.exit('USE rec_lensrec_RND0_cmbNG_fgG_partialNG.py for i==0')
else:
    cid1 = 'phiG'

if seed2==0:
    cid2 = 'phiNG'
    sys.exit('USE rec_lensrec_RND0_cmbNG_fgG_partialNG.py for i==0')
else:
    cid2 = 'phiG'

if comp1=='none': fid1 = 'tszG_kszG_cibG_radG'
if comp1=='tsz' : fid1 = 'tszNG_kszG_cibG_radG'
if comp1=='ksz' : fid1 = 'tszG_kszNG_cibG_radG'
if comp1=='cib' : fid1 = 'tszG_kszG_cibNG_radG'
if comp1=='rad' : fid1 = 'tszG_kszG_cibG_radNG'
if comp1=='all' : fid1 = 'tszNG_kszNG_cibNG_radNG'

if comp2=='none': fid2 = 'tszG_kszG_cibG_radG'
if comp2=='tsz' : fid2 = 'tszNG_kszG_cibG_radG'
if comp2=='ksz' : fid2 = 'tszG_kszNG_cibG_radG'
if comp2=='cib' : fid2 = 'tszG_kszG_cibNG_radG'
if comp2=='rad' : fid2 = 'tszG_kszG_cibG_radNG'
if comp2=='all' : fid2 = 'tszNG_kszNG_cibNG_radNG'

#-------- Loading map 1 ---------
if ilctype=='x':
    ilctype1 = 'cmbynull'
    ilctype2 = 'cibnull'

elif ilctype=='cmbmv':
    ilctype1 = 'cmbmv'
    ilctype2 = 'cmbmv'

elif ilctype=='mh':
    ilctype1 = 'cmbynull'
    ilctype2 = 'cmbmv'


print('loading:',dir_tmp+'/gaussmaps/cmb%d_%s_%s_%d.alm'%(cmbset1,ilctype1,cid1+"_"+fid1,seed1))
tlm1,elm1,blm1 = hp.read_alm(dir_tmp+'/gaussmaps/cmb%d_%s_%s_%d.alm'%(cmbset1,ilctype1,cid1+"_"+fid1,seed1),hdu=[1,2,3])

print('loading:',dir_tmp+'/gaussmaps/cmb%d_%s_%s_%d.alm'%(cmbset2,ilctype2,cid2+"_"+fid2,seed2))
tlm2,elm2,blm2 = hp.read_alm(dir_tmp+'/gaussmaps/cmb%d_%s_%s_%d.alm'%(cmbset2,ilctype2,cid2+"_"+fid2,seed2),hdu=[1,2,3])


#cltotres =np.loadtxt(dir_tmp+'/filter/filter_cls_fg_noise_%s.dat'%ilctype) 
cls_noise = np.loadtxt('./noise/cls_noise_%s.dat'%ilctype1)[:5101,1:5]
cls_totfg = np.loadtxt('./fgcls/cls_totfg_%s.dat'%ilctype1)[:5101,1:5]
cltotres1  = cls_totfg + cls_noise

cls_noise = np.loadtxt('./noise/cls_noise_%s.dat'%ilctype2)[:5101,1:5]
cls_totfg = np.loadtxt('./fgcls/cls_totfg_%s.dat'%ilctype2)[:5101,1:5]
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


Path(dir_tmp+'/lensrec_%s/'%ilctype).mkdir(parents=True, exist_ok=True)
np.savez(dir_tmp+'/lensrec_%s/plm%s_%d%s_%d%s_%s_x_%s.alm'%(ilctype,args.qetype,seed1,cmbset1name,seed2,cmbset2name,cid1+"_"+fid1,cid2+"_"+fid2),glm=glm,analytical_resp=respP)
print("done")





plm1      = np.load(file_plm1)
plm2      = np.load(file_plm2)
resp12    = resp_xest(qe1,qe2,cambcls,dict_cls,dict_lrange)
plm, resp = harden_est(plm1, plm2, resp12)

np.savez(file_plm ,glm=plm, analytical_resp=resp)



