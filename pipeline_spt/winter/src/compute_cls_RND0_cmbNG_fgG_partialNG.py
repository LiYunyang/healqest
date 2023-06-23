import os
import sys
import yaml
import pathlib
import numpy as np
import healpy as hp
import subprocess

file_yaml = str(sys.argv[1])
i         = int(sys.argv[2])
comp      = str(sys.argv[3])
qe        = str(sys.argv[4])
fgseed    = int(sys.argv[5])

params    = yaml.safe_load(open(file_yaml))
file_mask = params['masks']['dir_mask']+'mask2048_mdpl2_clusters_spt3g_10sigma_apod3.fits'
dir_base  = params['dirs']['dir_tmp']
    
def spice(qid,dir_cls,file_mask,dir_p,i,ktype='xxxx',comp='totfg',fgseed=0):
    os.environ['HEALPIX'] = "/lcrc/project/SPT3G/users/ac.yomori/envs/analysis/Healpix_3.80/"
    spice='/lcrc/project/SPT3G/users/ac.yomori/packages/PolSpice_v03-07-03/bin/spice'

    u=i+1
    if   ktype=='xxxx': ii=i; jj=i; xx=i; yy=i
    elif ktype=='xyxy': ii=i; jj=u; xx=i; yy=u
    elif ktype=='xyyx': ii=i; jj=u; xx=u; yy=i
    elif ktype=='x0x0': ii=i; jj=0; xx=i; yy=0
    elif ktype=='x00x': ii=i; jj=0; xx=0; yy=i
    elif ktype=='0xx0': ii=0; jj=i; xx=i; yy=0
    elif ktype=='0x0x': ii=0; jj=i; xx=0; yy=i

    file_1 = dir_p+'/kmap%s_%d.fits'%(ktype[:2],i)
    file_2 = dir_p+'/kmap%s_%d.fits'%(ktype[2:],i)

    subprocess.call([spice,'-mapfile'     , file_1,
                           '-weightfile'  , file_mask,
                           '-mapfile2'    , file_2,
                           '-weightfile2' , file_mask,
                           '-clfile'      , dir_cls+'clkk_%s_%da_%da_%da_%da_%s_%s_%s_%s_fgseed%d.dat'%(qid,ii,jj,xx,yy,comp,comp,comp,comp,fgseed),
                           '-nlmax'       ,'5100',
                           '-apodizesigma','170',
                           '-thetamax'    ,'180',
                           '-subav'       ,'YES',
                           '-verbosity'   , 'NO',
                        ])

def get_klm(dir_base,i,ktype='xx',qetype='TT',compname='all',fgseed=0):

    l=np.arange(5101)

    u=i+1
    if   ktype=='xx': ii='%da'%i; jj='%da'%i
    elif ktype=='xy': ii='%da'%u; jj='%da'%i
    elif ktype=='yx': ii='%da'%i; jj='%da'%u
    elif ktype=='x0': ii='%da'%i; jj='%da'%0
    elif ktype=='0x': ii='%da'%0; jj='%da'%i
    elif ktype=='ab': ii='%da'%i; jj='%db'%i
    elif ktype=='ba': ii='%db'%i; jj='%da'%i
    else: sys.exit('Undefined')
    
    if compname=='none': fid = 'phiNG_tszG_kszG_cibG_radG'
    if compname=='tsz' : fid = 'phiNG_tszNG_kszG_cibG_radG'
    if compname=='ksz' : fid = 'phiNG_tszG_kszNG_cibG_radG'
    if compname=='cib' : fid = 'phiNG_tszG_kszG_cibNG_radG'
    if compname=='rad' : fid = 'phiNG_tszG_kszG_cibG_radNG'
    if compname=='all' : fid = 'phiNG_tszNG_kszNG_cibNG_radNG'

    fidG = 'phiG_tszG_kszG_cibG_radG'
    fid0 = 'phiNG_tszNG_kszNG_cibNG_radNG'


    print("qetype:%s"%qetype)

    if qetype=='MV':
        qes = ['TT','EE','EB','TE','TB']

    elif qetype=='PP':
        qes = ['EE','EB']

    elif qetype=='TT' or qetype=='TE' or qetype=='TB' or qetype=='EB' or qetype=='EE':
        qes = [qetype]


    for qe in qes:
        print('using %s estimator'%qe)

     
    kmv    = 0
    respmv = 0

    for qe in qes:

        # Choose response
        resp = np.load(dir_base+'/lensrec/respavg%s.npz'%qe)['resp']
        resp[-100:]=np.inf

        # Load plm
        if ktype=='xx':
            if i==0:
                # all nonGaussian foreground
                mf = np.load(dir_base+'/lensrec/plmstack%s_%s.alm.npz'%(qe,ktype))
                y  = np.load(dir_base+'/lensrec_cmbNG_fgG_partialNG/fgseed%d/%s/plm%s_%s_%s_%s_x_%s_fgseed%d.alm.npz'%(fgseed,qe,qe,ii,jj,fid,fid,fgseed))['glm']
                k  = hp.almxfl(y-(mf['mf'])/(mf['nsim']),0.5*l*(l+1)*1/resp)
            else:
                # all Gaussian foreground
                mf = np.load(dir_base+'/lensrec/plmstack%s_%s.alm.npz'%(qe,ktype) )
                y  = np.load(dir_base+'/lensrec/plm%s_%s_%s_%s_x_%s.alm.npz'%(qe,ii,jj,fidG,fidG))['glm']
                k  = hp.almxfl(y-(mf['mf']-y)/(mf['nsim']-1.0),0.5*l*(l+1)*1/resp)

        elif ktype=='xy' or ktype=='yx':
            if i>0:
                mf = np.load(dir_base+'/lensrec/plmstack%s_%s.alm.npz'%(qe,ktype) )
                y  = np.load(dir_base+'/lensrec/plm%s_%s_%s_%s_x_%s.alm.npz'%(qe,ii,jj,fidG,fidG))['glm']
                k  = hp.almxfl(y-(mf['mf']-y)/(mf['nsim']-1.0),0.5*l*(l+1)*1/resp)
        
        elif ktype=='x0':
            if i>0:
                mf = np.load(dir_base+'/lensrec_cmbNG_fgG_partialNG/fgseed%d/%s/plmstack%s_%s_%s_x_%s_fgseed%d.alm.npz'%(fgseed,qe,qe,ktype,fidG,fid,fgseed))
                y  = np.load(dir_base+'/lensrec_cmbNG_fgG_partialNG/fgseed%d/%s/plm%s_%s_%s_%s_x_%s_fgseed%d.alm.npz'%(fgseed,qe,qe,ii,jj,fidG,fid,fgseed))['glm']
                k  = hp.almxfl(y-(mf['mf']-y)/(mf['nsim']-1.0),0.5*l*(l+1)*1/resp)
        elif ktype=='0x': # by definition i>0
            if i>0:
                mf = np.load(dir_base+'/lensrec_cmbNG_fgG_partialNG/fgseed%d/%s/plmstack%s_%s_%s_x_%s_fgseed%d.alm.npz'%(fgseed,qe,qe,ktype,fid,fidG,fgseed))
                y  = np.load(dir_base+'/lensrec_cmbNG_fgG_partialNG/fgseed%d/%s/plm%s_%s_%s_%s_x_%s_fgseed%d.alm.npz'%(fgseed,qe,qe,ii,jj,fid,fidG,fgseed))['glm']
                k  = hp.almxfl(y-(mf['mf']-y)/(mf['nsim']-1.0),0.5*l*(l+1)*1/resp)

        else: 
            print("specified ktype=%s"%ktype)
            sys.exit('aborting')


        kmv    += hp.almxfl(k,resp)
        respmv += resp

    respmv = 1/(respmv)
    respmv[-100:]=np.inf

    kmv     = hp.almxfl(kmv,respmv)
    ell,emm = hp.Alm.getlm(5100)
    kmv[ell>5000]=0

    return kmv

###################################################################
#dir_base = '/lcrc/globalscratch/ac.yomori/mdpl2_lensingbiases6/'
dir_p    = '/tmp/'
dir_cls  = dir_base+'/clkk_cmbNG_fgG_partialNG/%s/%s/run%d/'%(qe,comp,fgseed)

qid      = 'k%s'%qe.lower()

pathlib.Path(dir_cls).mkdir(parents=True, exist_ok=True)

if fgseed==0:
    if i==0:
        kxx = get_klm(dir_base,i,ktype='xx',qetype=qe,compname=comp,fgseed=fgseed); kmapxx = hp.alm2map(kxx,2048); hp.write_map(dir_p+'/kmapxx_%d.fits'%i,kmapxx,overwrite=True,dtype=np.float32)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='xxxx',comp=comp,fgseed=fgseed)
        os.remove(dir_p+'/kmapxx_%d.fits'%i)
    else:   
        kxx = get_klm(dir_base,i,ktype='xx',qetype=qe,compname=comp,fgseed=fgseed); kmapxx = hp.alm2map(kxx,2048); hp.write_map(dir_p+'/kmapxx_%d.fits'%i,kmapxx,overwrite=True,dtype=np.float32)
        kxy = get_klm(dir_base,i,ktype='xy',qetype=qe,compname=comp,fgseed=fgseed); kmapxy = hp.alm2map(kxy,2048); hp.write_map(dir_p+'/kmapxy_%d.fits'%i,kmapxy,overwrite=True,dtype=np.float32)
        kyx = get_klm(dir_base,i,ktype='yx',qetype=qe,compname=comp,fgseed=fgseed); kmapyx = hp.alm2map(kyx,2048); hp.write_map(dir_p+'/kmapyx_%d.fits'%i,kmapyx,overwrite=True,dtype=np.float32)
        kx0 = get_klm(dir_base,i,ktype='x0',qetype=qe,compname=comp,fgseed=fgseed); kmapx0 = hp.alm2map(kx0,2048); hp.write_map(dir_p+'/kmapx0_%d.fits'%i,kmapx0,overwrite=True,dtype=np.float32)
        k0x = get_klm(dir_base,i,ktype='0x',qetype=qe,compname=comp,fgseed=fgseed); kmap0x = hp.alm2map(k0x,2048); hp.write_map(dir_p+'/kmap0x_%d.fits'%i,kmap0x,overwrite=True,dtype=np.float32)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='xxxx',comp=comp,fgseed=fgseed)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='xyxy',comp=comp,fgseed=fgseed)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='xyyx',comp=comp,fgseed=fgseed)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='x0x0',comp=comp,fgseed=fgseed)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='x00x',comp=comp,fgseed=fgseed)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='0xx0',comp=comp,fgseed=fgseed)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='0x0x',comp=comp,fgseed=fgseed)
        os.remove(dir_p+'/kmapxx_%d.fits'%i)
        os.remove(dir_p+'/kmapxy_%d.fits'%i)
        os.remove(dir_p+'/kmapyx_%d.fits'%i)
        os.remove(dir_p+'/kmapx0_%d.fits'%i)
        os.remove(dir_p+'/kmap0x_%d.fits'%i)
else:
    if i==0:
        kxx = get_klm(dir_base,i,ktype='xx',qetype=qe,compname=comp,fgseed=fgseed); kmapxx = hp.alm2map(kxx,2048); hp.write_map(dir_p+'/kmapxx_%d.fits'%i,kmapxx,overwrite=True,dtype=np.float32)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='xxxx',comp=comp,fgseed=fgseed)
        os.remove(dir_p+'/kmapxx_%d.fits'%i)
    else:
        kx0 = get_klm(dir_base,i,ktype='x0',qetype=qe,fgseed=fgseed); kmapx0 = hp.alm2map(kx0,2048) ; hp.write_map(dir_p+'/kmapx0_%d.fits'%i,kmapx0,overwrite=True,dtype=np.float32)
        k0x = get_klm(dir_base,i,ktype='0x',qetype=qe,fgseed=fgseed); kmap0x = hp.alm2map(k0x,2048) ; hp.write_map(dir_p+'/kmap0x_%d.fits'%i,kmap0x,overwrite=True,dtype=np.float32)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='x0x0',comp=comp,fgseed=fgseed)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='x00x',comp=comp,fgseed=fgseed)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='0xx0',comp=comp,fgseed=fgseed)
        spice(qid,dir_cls,file_mask,dir_p,i,ktype='0x0x',comp=comp,fgseed=fgseed)
        os.remove(dir_p+'/kmapx0_%d.fits'%i)
        os.remove(dir_p+'/kmap0x_%d.fits'%i)
