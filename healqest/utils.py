import os,sys,yaml
import numpy as np
import healpy as hp
from pathlib import Path
import yaml,pickle
import logging as lg
logger = lg.getLogger(__name__)


def reduce_lmax(alm, lmax=4000):
    """
    Reduce the lmax of input alms (1d or 2d)
    """
    lmaxin = hp.Alm.getlmax(alm.shape[-1])
    logger.debug(f"Reducing lmax: lmax_in={lmaxin} -> lmax_out={lmax}")
    almout = np.zeros((*alm.shape[:-1], hp.Alm.getsize(lmax)), dtype=alm.dtype)
    oldi=0
    newi=0
    dl = lmaxin-lmax
    for i in range(0, lmax+1):
        oldf=oldi+lmaxin+1-i
        newf=newi+lmax+1-i
        almout[..., newi:newf]=alm[..., oldi:oldf-dl]
        oldi=oldf
        newi=newf
    return almout


def zeropad(cl):
    """add zeros for L=0,1"""
    cl=np.insert(cl,0,0)
    cl=np.insert(cl,0,0)
    return cl


def get_lensedcls(file,lmax=2000, dict=False):
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
    if dict==False:
        return ell,sltt,slee,slbb,slte
    else:
        d={}
        d['tt']=sltt
        d['ee']=slee
        d['bb']=slbb
        d['te']=slte
        return d

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
