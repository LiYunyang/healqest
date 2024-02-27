"""
Similar to opfilt_teb.py for flatsky, this is for QU/EB filtering for healpix
maps.
Modified from and built on plancklens/qcinv/opfilt_tp.py
"""

import os,sys
import hashlib
import numpy  as np
import healpy as hp

from healpy import alm2map_spin, map2alm_spin

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
import hp_utils
import cinv_utils

clhash = cinv_utils.clhash
teblm = hp_utils.teblm
_cli  = cinv_utils.cli

def calc_prep(maps, s_inv_filt, n_inv_filt):
    tmap, qmap, umap = np.copy(maps[0]), np.copy(maps[1]), np.copy(maps[2])
    assert (len(tmap) == len(qmap));
    assert (len(tmap) == len(umap))
    npix = len(tmap)

    n_inv_filt.apply_map([tmap, qmap, umap])
    lmax = len(n_inv_filt.b_transf) - 1

    tlm, elm, blm = hp.map2alm([tmap, qmap, umap], lmax=lmax, iter=0, pol=True)
    tlm *= npix / (4. * np.pi)
    elm *= npix / (4. * np.pi)
    blm *= npix / (4. * np.pi)

    if n_inv_filt.tf2d_t is None:
        hp.almxfl(tlm, n_inv_filt.b_transf_t, inplace=True)
        hp.almxfl(elm, n_inv_filt.b_transf_e, inplace=True)
        hp.almxfl(blm, n_inv_filt.b_transf_b, inplace=True)
    else:
        tlm *= n_inv_filt.tf2d_t
        elm *= n_inv_filt.tf2d_eb
        blm *= n_inv_filt.tf2d_eb

    return teblm([tlm, elm, blm])

def calc_fini(alm, s_inv_filt, n_inv_filt): 
    ret = s_inv_filt.calc(alm)
    alm.tlm[:] = ret.tlm[:]
    alm.elm[:] = ret.elm[:]
    alm.blm[:] = ret.blm[:]

class DotOperator:
    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        assert alm1.lmaxt == alm2.lmaxt, (alm1.lmaxt, alm2.lmaxt)
        assert alm1.lmaxe == alm2.lmaxe, (alm1.lmaxe, alm2.lmaxe)
        assert alm1.lmaxb == alm2.lmaxb, (alm1.lmaxb, alm2.lmaxb)

        ret =  np.sum(hp.alm2cl(alm1.tlm, alm2.tlm) * (2. * np.arange(0, alm1.lmaxt + 1) + 1))
        ret += np.sum(hp.alm2cl(alm1.elm, alm2.elm) * (2. * np.arange(0, alm1.lmaxe + 1) + 1))
        ret += np.sum(hp.alm2cl(alm1.blm, alm2.blm) * (2. * np.arange(0, alm1.lmaxb + 1) + 1))
        return ret

class ForwardOperator:
    def __init__(self, s_inv_filt, n_inv_filt):
        lmax = len(n_inv_filt.b_transf) - 1
        self.s_inv_filt = s_inv_filt #alm_filter_sinv(s_cls, lmax)
        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return {'s_inv_filt': self.s_inv_filt.hashdict(),
                'n_inv_filt': self.n_inv_filt.hashdict()}

    def __call__(self, alm):
        return self.calc(alm)

    def calc(self, alm):
        nlm = alm * 1.0
        self.n_inv_filt.apply_alm(nlm)
        slm = self.s_inv_filt.calc(alm)
        return nlm + slm

class PreOperatorDiag:
    def __init__(self, s_cls, n_inv_filt, n_cls=None):
        # diagonal pre-conditioner for conjugate-gradient solver
        # skip n_ell component (fine if white noise in ninv has most of the non-signal)
        lmax = len(n_inv_filt.b_transf) - 1
        s_inv_filt = alm_filter_sinv(s_cls, lmax)
        assert ((s_inv_filt.lmax + 1) >= len(n_inv_filt.b_transf))

        ninv_ftl, ninv_fel, ninv_fbl = n_inv_filt.get_ftebl()

        lmax = len(n_inv_filt.b_transf) - 1

        flmat = s_inv_filt.slinv[0:lmax + 1, :, :]

        flmat[:, 0, 0] += ninv_ftl
        flmat[:, 1, 1] += ninv_fel
        flmat[:, 2, 2] += ninv_fbl
        flmat = np.linalg.pinv(flmat)
        self.flmat = flmat
        self.te_only = s_inv_filt.te_only

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, alm):
        tmat = self.flmat
        if self.te_only:
            rtlm = hp.almxfl(alm.tlm, tmat[:, 0, 0]) + hp.almxfl(alm.elm, tmat[:, 0, 1])
            relm = hp.almxfl(alm.tlm, tmat[:, 1, 0]) + hp.almxfl(alm.elm, tmat[:, 1, 1])
            rblm = hp.almxfl(alm.blm, tmat[:, 2, 2])
        else:
            rtlm = hp.almxfl(alm.tlm, tmat[:, 0, 0]) + hp.almxfl(alm.elm, tmat[:, 0, 1]) + hp.almxfl(alm.blm, tmat[:, 0, 2])
            relm = hp.almxfl(alm.tlm, tmat[:, 1, 0]) + hp.almxfl(alm.elm, tmat[:, 1, 1]) + hp.almxfl(alm.blm, tmat[:, 1, 2])
            rblm = hp.almxfl(alm.tlm, tmat[:, 2, 0]) + hp.almxfl(alm.elm, tmat[:, 2, 1]) + hp.almxfl(alm.blm, tmat[:, 2, 2])
        return teblm([rtlm, relm, rblm])

class SkyInverseFilter: #alm_filter_sinv:
    def __init__(self, s_cls, lmax, n_cls=None, tf2d=None, tf2d_eb=None,
                 b_transf=None, b_transf_eb=None):
        '''
        tf2d, tf2d_eb, b_transf, b_transf_eb are TF in Cl (no rescal_cl factors)
        '''
        if n_cls is not None and tf2d is not None:
            #n_cls are 2d
            assert 'tt' in n_cls.keys()
            assert 'te' not in n_cls.keys()
            assert len(n_cls['tt']) == len(n_cls['ee']) == len(n_cls['bb'])== len(tf2d), "n_cls must be 2d (and TF+beam-ed)"
            if tf2d_eb is None:
                tf2d_eb = tf2d

            cltt = s_cls.get('tt', np.zeros(lmax + 1))[:lmax + 1]
            clte = s_cls.get('te', np.zeros(lmax + 1))[:lmax + 1]
            clee = s_cls.get('ee', np.zeros(lmax + 1))[:lmax + 1]
            clbb = s_cls.get('bb', np.zeros(lmax + 1))[:lmax + 1]

            cltt_2d = ytils.cl2almformat(cltt)
            clte_2d = ytils.cl2almformat(clte)
            clee_2d = ytils.cl2almformat(clee)
            clbb_2d = ytils.cl2almformat(clbb)

            slmat_alm = np.zeros((len(tf2d), 3,3))
            slmat_alm[:, 0, 0] = cltt_2d + n_cls['tt']*_cli(tf2d*tf2d)
            slmat_alm[:, 0, 1] = clte_2d
            slmat_alm[:, 1, 0] = slmat_alm[:, 0, 1]
            slmat_alm[:, 1, 1] = clee_2d + n_cls['ee']*_cli(tf2d_eb*tf2d_eb)
            slmat_alm[:, 2, 2] = clbb_2d + n_cls['bb']*_cli(tf2d_eb*tf2d_eb)

            self.slinv_alm = np.linalg.pinv(slmat_alm)

        if n_cls is not None:
            self.ncls1ds = self.ncls1d = {'tt': np.sqrt(hp.alm2cl(n_cls['tt'])),
                           'ee': np.sqrt(hp.alm2cl(n_cls['ee'])),
                           'bb': np.sqrt(hp.alm2cl(n_cls['bb']))}
        else:
            self.ncls1ds = {a: np.zeros(lmax+1) for a in s_cls.keys()}
            self.ncls1d = n_cls

        if b_transf is None:
            assert tf2d is None, "if tf2d is not None, nor should b_transf"
            b_transf = np.ones(lmax+1)
        if b_transf_eb is None: b_transf_eb = b_transf

        #always initialize diagonal (no l,m difference) S-inv; for PreOperatorDiag
        slmat = np.zeros((lmax + 1, 3, 3))  # matrix of TEB correlations at each l.
        slmat[:, 0, 0] = s_cls.get('tt', np.zeros(lmax + 1))[:lmax+1] + self.ncls1ds['tt']*_cli(b_transf*b_transf)
        slmat[:, 0, 1] = s_cls.get('te', np.zeros(lmax + 1))[:lmax+1]
        slmat[:, 1, 0] = slmat[:, 0, 1]
        slmat[:, 0, 2] = s_cls.get('tb', np.zeros(lmax + 1))[:lmax+1]
        slmat[:, 2, 0] = slmat[:, 0, 2]
        slmat[:, 1, 1] = s_cls.get('ee', np.zeros(lmax + 1))[:lmax+1] + self.ncls1ds['ee']*_cli(b_transf_eb*b_transf_eb)
        slmat[:, 1, 2] = s_cls.get('eb', np.zeros(lmax + 1))[:lmax+1]
        slmat[:, 2, 1] = slmat[:, 1, 2]
        slmat[:, 2, 2] = s_cls.get('bb', np.zeros(lmax + 1))[:lmax+1] + self.ncls1ds['bb']*_cli(b_transf_eb*b_transf_eb)
        slinv = np.linalg.pinv(slmat)

        self.lmax = lmax
        self.slinv = slinv
        self.s_cls = s_cls

        self.te_only = True
        if np.any(slmat[:, 0, 2]) or np.any(slmat[:, 1, 2]):
            self.te_only = False

        self.n_cls = n_cls
        self.tf2d  = tf2d
    def calc(self, alm):
        tmat = self.slinv

        if self.n_cls is not None and self.tf2d is not None:
            assert self.te_only, "only has te_only case"
            rtlm = alm.tlm * self.slinv_alm[:, 0, 0] + alm.elm * self.slinv_alm[:, 0, 1]
            relm = alm.tlm * self.slinv_alm[:, 1, 0] + alm.elm * self.slinv_alm[:, 1, 1]
            rblm = alm.blm * self.slinv_alm[:, 2, 2]
        else:
          if self.te_only:
            rtlm = hp.almxfl(alm.tlm, tmat[:, 0, 0]) + hp.almxfl(alm.elm, tmat[:, 0, 1])
            relm = hp.almxfl(alm.tlm, tmat[:, 1, 0]) + hp.almxfl(alm.elm, tmat[:, 1, 1])
            rblm = hp.almxfl(alm.blm, tmat[:, 2, 2])
          else:
            rtlm = hp.almxfl(alm.tlm, tmat[:, 0, 0]) + hp.almxfl(alm.elm, tmat[:, 0, 1]) + hp.almxfl(alm.blm, tmat[:, 0, 2])
            relm = hp.almxfl(alm.tlm, tmat[:, 1, 0]) + hp.almxfl(alm.elm, tmat[:, 1, 1]) + hp.almxfl(alm.blm, tmat[:, 1, 2])
            rblm = hp.almxfl(alm.tlm, tmat[:, 2, 0]) + hp.almxfl(alm.elm, tmat[:, 2, 1]) + hp.almxfl(alm.blm, tmat[:, 2, 2])
        return teblm([rtlm, relm, rblm])

    def hashdict(self):
        return {'slinv': clhash(self.slinv.flatten())}

class NoiseInverseFilter:
    def __init__(self, n_inv, b_transf, b_transf_e=None, b_transf_b=None,
                 tf2d=None, tf2d_eb=None,
                 #marge_monopole=False, marge_dipole=False, marge_maps_t=(), marge_maps_p=()
                ):
        '''
        tf2d, tf2d_eb, b_transf, b_transf_eb are TF in rescal_cl space
        '''
        # n_inv = [util.load_map(n[:]) for n in n_inv]
        self.n_inv = []
        for i, tn in enumerate(n_inv):
            if isinstance(tn, list):
                n_inv_prod = read_map(tn[0][:])
                if len(tn) > 1:
                    for n in tn[1:]:
                        n_inv_prod = n_inv_prod * read_map(n[:])
                self.n_inv.append(n_inv_prod)
                # assert (np.std(self.n_inv[i][np.where(self.n_inv[i][:] != 0.0)]) / np.average(
                #    self.n_inv[i][np.where(self.n_inv[i][:] != 0.0)]) < 1.e-7)
            else:
                self.n_inv.append(read_map(n_inv[i]))

        n_inv = self.n_inv
        npix = len(n_inv[0])
        nside = hp.npix2nside(npix)
        for n in n_inv[1:]:
            assert (len(n) == npix)
                self.npix = npix
        self.nside = nside


        self.b_transf_t = b_transf
        self.b_transf_e = b_transf_e if b_transf_e is not None else b_transf
        self.b_transf_b = b_transf_b if b_transf_b is not None else b_transf
        assert len(self.b_transf_t) == len(self.b_transf_e) and len(self.b_transf_t) == len(self.b_transf_e)
        self.b_transf = (self.b_transf_t + self.b_transf_e + self.b_transf_t) / 3.

        self.tf2d_t  = tf2d
        self.tf2d_eb = tf2d_eb if tf2d_eb is not None else tf2d

    def get_ftebl(self):
        if len(self.n_inv) == 2:  # TT, 1/2(QQ+UU)
            n_inv_cl_t = np.sum(self.n_inv[0]) / (4.0 * np.pi) * self.b_transf_t ** 2
            n_inv_cl_e = np.sum(self.n_inv[1]) / (4.0 * np.pi) * self.b_transf_e ** 2
            n_inv_cl_b = np.sum(self.n_inv[1]) / (4.0 * np.pi) * self.b_transf_b ** 2
            return n_inv_cl_t, n_inv_cl_e, n_inv_cl_b
        elif len(self.n_inv) == 4:  # TT, QQ, QU, UU
            n_inv_cl_t = np.sum(self.n_inv[0]) / (4.0 * np.pi) * self.b_transf_t ** 2
            n_inv_cl_e = np.sum(0.5 * (self.n_inv[1] + self.n_inv[3])) / (4.0 * np.pi) * self.b_transf_e ** 2
            n_inv_cl_b = np.sum(0.5 * (self.n_inv[1] + self.n_inv[3])) / (4.0 * np.pi) * self.b_transf_b ** 2

            return n_inv_cl_t, n_inv_cl_e, n_inv_cl_b
        else:
            assert 0

    def hashdict(self):
        return {'n_inv': [clhash(n) for n in self.n_inv],
                'b_transf': clhash(self.b_transf),
                }

    def apply_alm(self, alm):
        # applies Y^T N^{-1} Y
        lmax = alm.lmax

        if self.tf2d_t is None:
            hp.almxfl(alm.tlm, self.b_transf_t, inplace=True)
            hp.almxfl(alm.elm, self.b_transf_e, inplace=True)
            hp.almxfl(alm.blm, self.b_transf_b, inplace=True)
        else:
            alm.tlm *= self.tf2d_t
            alm.elm *= self.tf2d_eb
            alm.blm *= self.tf2d_eb
        tmap, qmap, umap = hp.alm2map((alm.tlm, alm.elm, alm.blm), self.nside, pol=True)
        # qmap, umap = hp.alm2map_spin((alm.elm, alm.blm), self.nside, 2)

        self.apply_map([tmap, qmap, umap])

        ttlm, telm, tblm = hp.map2alm([tmap, qmap, umap], iter=0, pol=True, lmax=lmax)
        alm.tlm[:] = ttlm
        alm.elm[:] = telm
        alm.blm[:] = tblm

        alm.tlm[:] *= (self.npix / (4. * np.pi))
        alm.elm[:] *= (self.npix / (4. * np.pi))
        alm.blm[:] *= (self.npix / (4. * np.pi))

        if self.tf2d_t is None:
            hp.almxfl(alm.tlm, self.b_transf_t, inplace=True)
            hp.almxfl(alm.elm, self.b_transf_e, inplace=True)
            hp.almxfl(alm.blm, self.b_transf_b, inplace=True)
        else:
            alm.tlm *= self.tf2d_t
            alm.elm *= self.tf2d_eb
            alm.blm *= self.tf2d_eb

    def apply_map(self, amap):
        [tmap, qmap, umap] = amap

        # applies N^{-1}
        if len(self.n_inv) == 2:  # TT, QQ=UU
            tmap *= self.n_inv[0]
            qmap *= self.n_inv[1]
            umap *= self.n_inv[1]
        elif len(self.n_inv) == 4:  # TT, QQ, QU, UU
            qmap_copy = qmap.copy()

            tmap *= self.n_inv[0]
            qmap *= self.n_inv[1]
            qmap += self.n_inv[2] * umap

            umap *= self.n_inv[3]
            umap += self.n_inv[2] * qmap_copy

            del qmap_copy
        else:
            assert 0