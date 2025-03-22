"""
Similar to opfilt_teb.py for flatsky, this is for QU/EB filtering for healpix
maps.

Modified from and built on plancklens/qcinv/opfilt_pp.py
"""

import os,sys
import numpy  as np
import healpy as hp

from healpy import alm2map_spin, map2alm_spin

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
import hp_utils
import cinv_utils

clhash = cinv_utils.clhash
eblm   = hp_utils.eblm

def calc_prep(maps, s_inv_filt, n_inv_filt):
    qmap = np.copy(hp_utils.read_map(maps[0]))
    umap = np.copy(hp_utils.read_map(maps[1]))
    assert len(qmap) == len(umap)
    lmax = len(n_inv_filt.b_transf) - 1
    npix = len(qmap)

    n_inv_filt.apply_map([qmap, umap])

    elm, blm = map2alm_spin([qmap, umap], 2, lmax=lmax)

    if n_inv_filt.tf2d is None:
        '''
        hp.almxfl(elm, n_inv_filt.b_transf_e * npix / (4. * np.pi), inplace=True)
        hp.almxfl(blm, n_inv_filt.b_transf_b * npix / (4. * np.pi), inplace=True)
        '''
        sys.exit('bad path')
    else:
        elm *= n_inv_filt.tf2d * npix / (4. * np.pi)
        blm *= n_inv_filt.tf2d * npix / (4. * np.pi)

    return eblm([elm, blm])

def calc_fini(alm, s_inv_filt, n_inv_filt):
    ret = s_inv_filt.calc(alm)
    alm.elm[:] = ret.elm[:]
    alm.blm[:] = ret.blm[:]

class DotOperator:
    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        assert alm1.lmax == alm2.lmax
        tcl = hp.alm2cl(alm1.elm, alm2.elm) + hp.alm2cl(alm1.blm, alm2.blm)
        return np.sum(tcl[2:] * (2. * np.arange(2, alm1.lmax + 1) + 1))

class ForwardOperator:
    """Missing doc. """
    def __init__(self, s_inv_filt, n_inv_filt):
        self.s_inv_filt = s_inv_filt
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
    """Missing doc. """
    def __init__(self, s_cls, n_inv_filt, n_cls=None):
        '''
        s_cls     : signal only theory Cls (dictionary with keys 'tt/'ee'/'bb')
        n_inv_filt: 
        n_cls     : |nlm|^2 (dictionary with keys 'tt/'ee'/'bb')
        '''
        lmax = len(n_inv_filt.b_transf) - 1
        clbb = s_cls['bb'][:lmax + 1]
        clee = s_cls['ee'][:lmax + 1]
        
        if n_cls is None:
            n_cls = {key: np.zeros(lmax+1) for key in s_cls}

        ninv_fel, ninv_fbl = n_inv_filt.get_febl()

        #import pdb; pdb.set_trace()

        # n_cls['ee'] = ncls1d  where ncls1d = {'ee': np.sqrt(hp.alm2cl(n_cls['e']))}
        tfbl   = hp.alm2cl(n_inv_filt.tf2d.astype(np.complex_) )[:lmax+1]
        filt_e = cinv_utils.cli(clee + n_cls['ee'] * cinv_utils.cli(tfbl ** 2))
        filt_e += ninv_fel[:lmax+1]

        filt_b = cinv_utils.cli(clbb + n_cls['bb'] * cinv_utils.cli(tfbl ** 2))
        filt_b += ninv_fbl[:lmax+1]

        self.filt_e = cinv_utils.cli(filt_e)
        self.filt_b = cinv_utils.cli(filt_b)

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, alm):
        relm = hp.almxfl(alm.elm, self.filt_e)
        rblm = hp.almxfl(alm.blm, self.filt_b)
        return eblm([relm, rblm])

class SkyInverseFilter: #alm_filter_sinv_nocorr:
    def __init__(self, s_cls, lmax, n_cls=None, tf2d=None, b_transf=None):

        self.n_cls = n_cls  # |nlm^2|
        
        if n_cls is not None and tf2d is not None:
            #print(len(n_cls['ee']))
            #print(len(tf2d))
            #the only case that needs to expand into alms
            assert 'ee' in n_cls.keys()
            assert 'bb' in n_cls.keys()
            assert len(n_cls['ee']) == len(tf2d), "n_cls must be 2d (and TF+beam-ed)"
            assert len(n_cls['bb']) == len(tf2d), "n_cls must be 2d (and TF+beam-ed)"

            clee = s_cls.get('ee', np.zeros(lmax + 1))[:lmax + 1]
            clbb = s_cls.get('bb', np.zeros(lmax + 1))[:lmax + 1]

            clee_2d = hp_utils.cl2almformat(clee)
            clbb_2d = hp_utils.cl2almformat(clbb)

            #self.b_slinv = utils.cli(clbb_2d + n_cls['ee'] *utils.cli(tf2d*tf2d))
            #self.e_slinv = utils.cli(clee_2d + n_cls['bb'] *utils.cli(tf2d*tf2d))
            self.b_slinv = tf2d * tf2d * cinv_utils.cli(clbb_2d * tf2d * tf2d + n_cls['bb'])
            self.e_slinv = tf2d * tf2d * cinv_utils.cli(clee_2d * tf2d * tf2d + n_cls['ee'])
        else:
            '''
            clee = s_cls.get('ee', np.zeros(lmax + 1))[:lmax + 1]
            clbb = s_cls.get('bb', np.zeros(lmax + 1))[:lmax + 1]
            if n_cls is None: n_cls = {key: np.zeros(lmax+1) for key in s_cls}
            nlee = n_cls.get('ee', np.zeros(lmax + 1))[:lmax + 1]
            nlbb = n_cls.get('bb', np.zeros(lmax + 1))[:lmax + 1]
            if b_transf is None:
                assert tf2d is None, "if tf2d is not None, nor should b_transf"
                b_transf = np.ones(lmax+1)
            self.b_slinv = cinv_utils.cli(clbb + nlbb*cinv_utils.cli(b_transf*b_transf))
            self.e_slinv = cinv_utils.cli(clee + nlee*cinv_utils.cli(b_transf*b_transf))
            '''
            sys.exit('bad path')

        self.lmax  = lmax
        self.s_cls = s_cls
        self.tf2d  = tf2d
        #import pdb; pdb.set_trace()
        if self.n_cls is not None and tf2d is not None:
            self.ncls1d = {'ee': np.sqrt(hp.alm2cl(n_cls['ee'])),
                           'bb': np.sqrt(hp.alm2cl(n_cls['bb']))}
        else:
            self.ncls1d = self.n_cls

    def calc(self, alm):
        if self.n_cls is not None and self.tf2d is not None:
            relm = alm.elm * self.e_slinv # elm * 1/(clee+nlee/tf**2) 
            rblm = alm.blm * self.b_slinv # blm * 1/(clbb+nlbb/tf**2)
        else:
            '''
            relm = hp.almxfl(alm.elm, self.e_slinv, inplace=False)
            rblm = hp.almxfl(alm.blm, self.b_slinv, inplace=False)
            '''
            sys.exit('bad path')

        return eblm([relm, rblm])

    def hashdict(self):
        return {'b_slinv':clhash(self.b_slinv),
                'e_slinv':clhash(self.e_slinv)}

class NoiseInverseFilter:
    def __init__(self, n_inv, b_transf, tf2d=None,
                 nlev_febl=None, b_transf_b=None):
                 #, marge_qmaps=(), marge_umaps=()):
        """Inverse-variance filtering instance for polarization only

            Args:
                n_inv: inverse pixel variance maps or masks
                b_transf: filter fiducial transfer function
                nlev_febl(optional): isotropic approximation to the noise level across the entire map
                                     this is used e.g. in the diag. preconditioner of cg inversion.
                b_transf_b: B-mode transfer func if different from E-mode

            Note:
                This allows for independent Q and U map marginalization

        """

        self.b_transf_e = b_transf
        self.b_transf_b = b_transf_b if b_transf_b is not None else b_transf
        self.b_transf   = 0.5 * (self.b_transf_e + self.b_transf_b)
        self.tf2d       = tf2d

        # These three things will be instantiated later on
        self.nside = None
        self.n_inv = None

        self.nlev_febl = nlev_febl
        self._n_inv = n_inv # could be paths or list of paths
        self._load_ninv()

    def _load_ninv(self):
        print("==_load_ninv==")
        if self.n_inv is None:
            self.n_inv = []
            for i, tn in enumerate(self._n_inv):
                if isinstance(tn, list):
                    n_inv_prod = hp_utils.read_map(tn[0])
                    if len(tn) > 1:
                        for n in tn[1:]:
                            n_inv_prod = n_inv_prod * hp_utils.read_map(n)
                    self.n_inv.append(n_inv_prod)
                else:
                    self.n_inv.append(hp_utils.read_map(self._n_inv[i]))
            assert len(self.n_inv) in [1, 3], len(self.n_inv)
            self.nside = hp.npix2nside(len(self.n_inv[0]))

    def _calc_febl(self):
        print("==_calc_febl==")
        self._load_ninv()
        if len(self.n_inv) == 1:
            nlev_febl = 10800. / np.sqrt(np.sum(self.n_inv[0]) / (4.0 * np.pi)) / np.pi
        elif len(self.n_inv) == 3:
            nlev_febl = 10800. / np.sqrt(np.sum(0.5 * (self.n_inv[0] + self.n_inv[2])) / (4.0 * np.pi)) / np.pi
        else:
            assert 0
        print("ninv_febl: using %.2f uK-amin noise Cl"%nlev_febl)
        return nlev_febl

    def get_ninv(self):
        print("==get_ninv==")
        self._load_ninv()
        return self.n_inv

    def get_mask(self):
        print("==get_mask==")
        ninv = self.get_ninv()
        assert len(ninv) in [1, 3], len(ninv)
        self.nside = hp.npix2nside(len(ninv[0]))
        mask = np.where(ninv[0] > 0, 1., 0)
        for ni in ninv[1:]:
            mask *= (ni > 0)
        return mask

    def get_febl(self):
        print("==get_febl==")
        if self.nlev_febl is None:
            self.nlev_febl = self._calc_febl()
        n_inv_cl_e = self.b_transf_e ** 2  / (self.nlev_febl / 180. / 60. * np.pi) ** 2
        n_inv_cl_b = self.b_transf_b ** 2  / (self.nlev_febl / 180. / 60. * np.pi) ** 2
        return n_inv_cl_e, n_inv_cl_b

    def hashdict(self):
        ret = {'n_inv': [cinv_util.mask_hash(n, dtype=np.float16) for n in self._n_inv],
               'b_transf': clhash(self.b_transf)}
        return ret

    def apply_alm(self, alm):
        """B^dagger N^{-1} B"""
        print("apply_alm")
        self._load_ninv()
        lmax = alm.lmax

        if self.tf2d is None:
            #hp.almxfl(alm.elm, self.b_transf_e, inplace=True)
            #hp.almxfl(alm.blm, self.b_transf_b, inplace=True)
            sys.exit('no tf2d supplied')
        else:
            alm.elm *= self.tf2d
            alm.blm *= self.tf2d

        qmap, umap = alm2map_spin((alm.elm, alm.blm), self.nside, 2, lmax)

        self.apply_map([qmap, umap])  # applies N^{-1}
        npix = len(qmap)

        telm, tblm = map2alm_spin([qmap, umap], 2, lmax=lmax)
        alm.elm[:] = telm
        alm.blm[:] = tblm

        if self.tf2d is None:
            #hp.almxfl(alm.elm, self.b_transf_e * (npix / (4. * np.pi)), inplace=True)
            #hp.almxfl(alm.blm, self.b_transf_b * (npix / (4. * np.pi)), inplace=True)
            sys.exit('no tf2d supplied')
        else:
            alm.elm *= self.tf2d * (npix / (4. * np.pi))
            alm.blm *= self.tf2d * (npix / (4. * np.pi))

    def apply_map(self, amap):
        print("apply_map")
        self._load_ninv()
        [qmap, umap] = amap
        if len(self.n_inv) == 1:  # TT, QQ=UU
            print('only detected one n_inv map')
            qmap *= self.n_inv[0]
            umap *= self.n_inv[0]

        elif len(self.n_inv) == 3:  # TT, QQ, QU, UU
            print('detected three n_inv map')
            qmap_copy = qmap.copy()

            qmap *= self.n_inv[0]
            qmap += self.n_inv[1] * umap

            umap *= self.n_inv[2]
            umap += self.n_inv[1] * qmap_copy

            del qmap_copy
        else:
            assert 0






