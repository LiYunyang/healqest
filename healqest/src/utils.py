import numpy as np
import healpy as hp
from pathlib import Path
import logging 

def reduce_lmax(alm, lmax=4000):
    """
    Reduce the lmax of input alm
    """
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

def get_nside(lmax):
    """calculates the most appropriate nside based on lmax"""
    nside = np.array([8,16,32,64,128,256,512,1024,2048,4096,8192,16384])
    idx   = np.argmin(np.abs(nside-lmax))
    return nside[idx]

def zeropad(cl):
    """add zeros for L=0,1"""
    cl=np.insert(cl,0,0)
    cl=np.insert(cl,0,0)
    return cl

def get_lensedcls(file,lmax=2000):
    ell,sltt,slee,slbb,slte=np.loadtxt(file,unpack=True)
    # Removing the ell factors and padding with zeros (since the file starts with l=2)
    sltt=sltt/ell/(ell+1)*2*np.pi; sltt=zeropad(sltt)
    slee=slee/ell/(ell+1)*2*np.pi; slee=zeropad(slee)
    slte=slte/ell/(ell+1)*2*np.pi; slte=zeropad(slte)
    slbb=slbb/ell/(ell+1)*2*np.pi; slbb=zeropad(slbb)
    ell=np.insert(ell,0,1); ell=np.insert(ell,0,0)
    ell  = ell[:lmax+1]
    sltt = sltt[:lmax+1]
    slee = slee[:lmax+1]
    slbb = slbb[:lmax+1]
    slte = slte[:lmax+1]
    return ell,sltt,slee,slbb,slte 

def get_unlensedcls(file,lmax=2000):
    ell,sltt,slee,slbb,slte,slpp,sltp,slep=np.loadtxt(file,unpack=True)
    # Removing the ell factors and padding with zeros (since the file starts with l=2)
    sltt=sltt/ell/(ell+1)*2*np.pi;             sltt=zeropad(sltt)
    slee=slee/ell/(ell+1)*2*np.pi;             slee=zeropad(slee)
    slbb=slbb/ell/(ell+1)*2*np.pi;             slbb=zeropad(slbb)
    slte=slte/ell/(ell+1)*2*np.pi;             slte=zeropad(slte)
    slpp=slpp/ell/ell/(ell+1)/(ell+1)*2*np.pi; slpp=zeropad(slpp)
    sltp=sltp/(ell*(ell+1))**(1.5)*2*np.pi;    sltp=zeropad(sltp)
    slep=slep/(ell*(ell+1))**(1.5)*2*np.pi;    slep=zeropad(slep)
    ell=np.insert(ell,0,1); ell=np.insert(ell,0,0)
    ell  = ell[:lmax+1]
    sltt = sltt[:lmax+1]
    slee = slee[:lmax+1]
    slbb = slbb[:lmax+1]
    slte = slte[:lmax+1]
    slpp = slpp[:lmax+1]
    sltp = sltp[:lmax+1]
    slep = slep[:lmax+1]
    return ell,sltt,slee,slbb,slte,slpp,sltp,slep
 
def setup_logger(nolog,file_log='test.log'):

    dir_log = str(Path(file_log).parent)
    Path(dir_log).mkdir(parents=True, exist_ok=True)

    if nolog==True:
        logging.basicConfig(
                            format   = '[%(asctime)s] %(message)s',
                            datefmt  = '%H:%M:%S',
                            level    = logging.WARNING)
    else:
        logging.basicConfig(filename = file_log ,
                            filemode = 'w+',
                            format   = '[%(asctime)s] %(message)s',
                            datefmt  = '%H:%M:%S',
                            level    = logging.WARNING)

def add_clsdict(d,key,cltt,clee,clbb,clte=None):
    d[key]  = {}
    d[key]['tt'] = cltt
    d[key]['ee'] = clee
    d[key]['bb'] = clbb

    if clte is not None:
        d[key]['te'] = clte

    return d

def get_fl(cls, dict_lrange):
    lmaxTP = dict_lrange['lmaxTP']
    lmin   = dict_lrange['lmin']
    lmaxT  = dict_lrange['lmaxT']
    lmaxP  = dict_lrange['lmaxP']

    flT = 1.0/(cls['cmb']['tt']+cls['res']['tt'][:lmaxTP+1]); flT[lmaxT+1:] = 0; flT[:lmin] = 0
    flE = 1.0/(cls['cmb']['ee']+cls['res']['ee'][:lmaxTP+1]); flE[lmaxP+1:] = 0; flE[:lmin] = 0
    flB = 1.0/(cls['cmb']['bb']+cls['res']['bb'][:lmaxTP+1]); flB[lmaxP+1:] = 0; flB[:lmin] = 0

    return flT, flE, flB

def get_almbar(qetype, alms1, alms2, cls, dict_lrange):
    tlm1,elm1,blm1 = alms1[0],alms1[1],alms1[2]
    tlm2,elm2,blm2 = alms2[0],alms2[1],alms2[2]
    flT,flE,flB    = get_fl(cls, dict_lrange)

    lmaxTP = dict_lrange['lmaxTP']

    if qetype[0]=='T': tlm1 = reduce_lmax(tlm1,lmax=lmaxTP); almbar1 = hp.almxfl(tlm1,flT); flm1= flT
    if qetype[0]=='E': elm1 = reduce_lmax(elm1,lmax=lmaxTP); almbar1 = hp.almxfl(elm1,flE); flm1= flE
    if qetype[0]=='B': blm1 = reduce_lmax(blm1,lmax=lmaxTP); almbar1 = hp.almxfl(blm1,flB); flm1= flB
    if qetype[1]=='T': tlm2 = reduce_lmax(tlm2,lmax=lmaxTP); almbar2 = hp.almxfl(tlm2,flT); flm2= flT
    if qetype[1]=='E': elm2 = reduce_lmax(elm2,lmax=lmaxTP); almbar2 = hp.almxfl(elm2,flE); flm2= flE
    if qetype[1]=='B': blm2 = reduce_lmax(blm2,lmax=lmaxTP); almbar2 = hp.almxfl(blm2,flB); flm2= flB
    return almbar1,almbar2,flm1,flm2
