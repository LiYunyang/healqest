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
#sys.path.append('/lcrc/pr

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml', default=None, type=str, help='dir_base')
parser.add_argument('ilctype'  , default=None, type=str, help='ilctype')
parser.add_argument('nsims'    , default=10  , type=int, help='comp')
parser.add_argument('--getxx' , default=False, dest='getxx' ,action='store_true')
parser.add_argument('--getxy' , default=False, dest='getxy' ,action='store_true')
parser.add_argument('--getyx' , default=False, dest='getyx' ,action='store_true')
parser.add_argument('--get0x' , default=False, dest='get0x' ,action='store_true')
parser.add_argument('--getx0' , default=False, dest='getx0' ,action='store_true')
args = parser.parse_args()

nsims=args.nsims
getxx=args.getxx
getxy=args.getxy
getyx=args.getyx
get0x=args.get0x
getx0=args.getx0

#  = args.yaml
ilctype   = args.ilctype

params    = yaml.safe_load(open(args.file_yaml))
dir_tmp   = params['dirs']['dir_tmp']

#-----------------------------------------------------------------------

dir_p = dir_tmp+ '/lensrec_%s/' %ilctype

fidxx   = 'phiG_tszG_kszG_cibG_radG_x_phiG_tszG_kszG_cibG_radG'
fidxy   = 'phiG_tszG_kszG_cibG_radG_x_phiG_tszG_kszG_cibG_radG'
fidyx   = 'phiG_tszG_kszG_cibG_radG_x_phiG_tszG_kszG_cibG_radG'

l=np.arange(5101)

for qe in (['TT','EE','TE','TB','EB']):
    print("Processing %s"%qe)
    mfxx=0; mfxy=0; mfyx=0; mfab=0; mfba=0; mf0x=0; mfx0=0

    cc=0    
    for i in range(1,nsims+1):
        if getxx: plmxx=np.load(dir_p+'/plm%s_%da_%da_%s.alm.npz'%(qe,i,i,fidxx))['glm']; mfxx+=plmxx
        if getxy: plmxy=np.load(dir_p+'/plm%s_%da_%da_%s.alm.npz'%(qe,i,i+1,fidxy))['glm']; mfxy+=plmxy
        if getyx: plmyx=np.load(dir_p+'/plm%s_%da_%da_%s.alm.npz'%(qe,i+1,i,fidyx))['glm']; mfyx+=plmyx
        if get0x: plm0x=np.load(dir_p+'/plm%s_%da_%da_%s.alm.npz'%(qe,0,i,fid0x))['glm']; mf0x+=plm0x
        if getx0: plmx0=np.load(dir_p+'/plm%s_%da_%da_%s.alm.npz'%(qe,i,0,fidx0))['glm']; mfx0+=plmx0
           
        cc+=1

    if getxx: np.savez(dir_p+'plmstack%s_xx.alm'%(qe),mf=mfxx,nsim=cc)
    if getxy: np.savez(dir_p+'plmstack%s_xy.alm'%(qe),mf=mfxy,nsim=cc)
    if getyx: np.savez(dir_p+'plmstack%s_yx.alm'%(qe),mf=mfyx,nsim=cc)
    if get0x: np.savez(dir_p+'plmstack%s_0x.alm'%(qe),mf=mf0x,nsim=cc)
    if getx0: np.savez(dir_p+'plmstack%s_x0.alm'%(qe),mf=mfx0,nsim=cc)


for qe in (['TT','EE','TE','TB','EB']):
    print("Processing %s"%qe)

    cl1=0
    cl2=0

    c=0
    for i in range(1,nsims+1):
        ilm=np.load('/lcrc/globalscratch/ac.yomori/mdpl2_lensingbiases6/inputkappa/inputklm_nomask_seed%d.npy'%i)
        olm=np.load(dir_p+'/plm%s_%da_%da_%s.alm.npz'%(qe,i,i,fidxx))['glm'] 
        cl1+=hp.alm2cl(ilm,ilm)
        cl2+=hp.alm2cl(olm,ilm)*0.5*(l*(l+1))
        c+=1

    respavg=cl2/cl1

    respavg[:4]=1e30
    respavg[-1]=1e30

    np.savez(dir_p+'/respavg%s.npz'%qe,resp=respavg,nsim=c)

