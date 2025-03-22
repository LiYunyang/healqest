"""
Convinient functions for curved-sky lensing
"""

import healpy as hp
import numpy as np

def almtf2d(lmax, lmin=0, mmin=0, bl=None):
    """
    bl: 1D B(ell), start at ell=0    
    """

    tf2d = np.zeros([lmax+1,lmax+1])
    #fill non-zero (l,m) with ones
    for l in range(0,lmax+1):
        tf2d[0:l+1 , l] = 1.0
    if bl is not None:
        tf2d *= bl[None, :]

    tf2d[:mmin,:] = 0
    tf2d[:,:lmin] = 0

    return tf2d

def cl2almformat(cl):
    """
    repeat Cl for all m-modes at each ell
    return alm-ordering array
    cl array starts with ell=0
    """ 
    lmax = len(cl)-1
    alm = np.zeros(hp.Alm.getsize(lmax))
    idx = 0
    for i in range(0, lmax+1):
        alm[idx:idx+(lmax+1-i)] = cl[i:]
        idx = idx+(lmax+1-i)
    return alm   

def grid2alm(grid):
    """
    Convert 2d grid back to Healpix alm
    """
    lmax = grid.shape[0]-1
    alm=np.zeros(hp.Alm.getsize(lmax),dtype=np.complex_)
    for l in range(0,lmax+1):
        for m in range(0,l+1):
            # l,m
            alm[hp.Alm.getidx(lmax,l,m)]=grid[m,l]
            #alm[hp.Alm.getidx(lmax,i,np.arange(i+1)-1)]=grid[:i+1,i]
    return alm

def alm2grid(alm, realpart=True):
    """
    Convert Healpix alm array to 2d grid
    """

    #lmax = hp.Alm.getlmax(alm.shape[0])
    #ell,emm=hp.Alm.getlm(lmax)
    #grid=np.zeros((lmax+1,lmax+1),dtype=np.complex_)
    #for i in range(0,lmax+1):
    #        grid[:i+1,i]=alm[ell==i]
    #return grid

    lmax = hp.Alm.getlmax(alm.shape[0])
    ell,emm = hp.Alm.getlm(lmax)
    alm = np.abs(alm) if realpart else alm
    a=np.zeros((lmax+1,lmax+1), dtype=alm.dtype)
    idx  = 0
    idxf = 0
    for i in range(0,lmax+1):
        idxf = idx + lmax + 1 -i
        a[i,i:]=alm[idx:idx+(lmax+1-i)]
        idx=idxf
    return a

class eblm:
    def __init__(self, alm):
        [elm, blm] = alm
        assert len(elm) == len(blm), (len(elm), len(blm))

        self.lmax = hp.Alm.getlmax(len(elm))
        
        self.elm = elm
        self.blm = blm

    #def alm_copy(self, lmax=None):
    #    return eblm([alm_copy(self.elm, lmax=lmax),
    #                 alm_copy(self.blm, lmax=lmax)])

    def __add__(self, other):
        assert self.lmax == other.lmax
        return eblm([self.elm + other.elm, self.blm + other.blm])

    def __sub__(self, other):
        assert self.lmax == other.lmax
        return eblm([self.elm - other.elm, self.blm - other.blm])

    def __iadd__(self, other):
        assert self.lmax == other.lmax
        self.elm += other.elm
        self.blm += other.blm
        return self

    def __isub__(self, other):
        assert self.lmax == other.lmax
        self.elm -= other.elm
        self.blm -= other.blm
        return self

    def __mul__(self, other):
        return eblm([self.elm * other, self.blm * other])

def read_map(m):
    """Reads a map whether given as (list of) string (with ',f' denoting field f), array or callable        
    """
    if callable(m):
        return m()
    if isinstance(m, list):
        ma = read_map(m[0])
        for m2 in m[1:]:
            ma *= read_map(m2)
        return ma
    if not isinstance(m, str):
        return m
    if ',' not in m:
        return hp.read_map(m)
    m, field = m.split(',')
    return hp.read_map(m, field=int(field))

class jit:
    """ just-in-time instantiation wrapper class.

    """
    def __init__(self, ctype, *cargs, **ckwds):
        self.__dict__['__jit_args'] = [ctype, cargs, ckwds]
        self.__dict__['__jit_obj'] = None

    def instantiate(self):
        [ctype, cargs, ckwds] = self.__dict__['__jit_args']
        print('jit: instantiating ctype =', ctype)
        self.__dict__['__jit_obj'] = ctype(*cargs, **ckwds)
        del self.__dict__['__jit_args']

    def __getattr__(self, attr):
        if self.__dict__['__jit_obj'] is None:
            self.instantiate()
        return getattr(self.__dict__['__jit_obj'], attr)

    def __setattr__(self, attr, val):
        if self.__dict__['__jit_obj'] is None:
            self.instantiate()
        setattr(self.__dict__['__jit_obj'], attr, val)



