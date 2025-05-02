"""
Similar to opfilt_teb.py for flatsky, this is for joint TQU/TEB filtering for healpix
maps.

Modified from and built on plancklens/qcinv/opfilt_tp.py
"""

import numpy as np
import healpy as hp

from . import hp_utils, cinv_utils

clhash = cinv_utils.clhash
teblm = hp_utils.teblm
_cli = cinv_utils.cli


def calc_prep(maps, s_inv_filt, n_inv_filt):
    tmap, qmap, umap = np.copy(maps[0]), np.copy(maps[1]), np.copy(maps[2])
    assert (len(tmap) == len(qmap))
    assert (len(tmap) == len(umap))
    lmax = len(n_inv_filt.tf1d_t) - 1
    npix = len(tmap)

    n_inv_filt.apply_map([tmap, qmap, umap])

    tlm, elm, blm = hp.map2alm([tmap, qmap, umap], lmax=lmax, iter=0, pol=True)
    tlm *= npix / (4. * np.pi)
    elm *= npix / (4. * np.pi)
    blm *= npix / (4. * np.pi)

    if n_inv_filt.tf2d_t is None:
        hp.almxfl(tlm, n_inv_filt.tf1d_t, inplace=True)
        hp.almxfl(elm, n_inv_filt.tf1d_p, inplace=True)
        hp.almxfl(blm, n_inv_filt.tf1d_p, inplace=True)
    else:
        tlm *= n_inv_filt.tf2d_t
        elm *= n_inv_filt.tf2d_p
        blm *= n_inv_filt.tf2d_p

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
        # CHECKED WITH PLANCK LENSING
        assert alm1.lmaxt == alm2.lmaxt, (alm1.lmaxt, alm2.lmaxt)
        assert alm1.lmaxe == alm2.lmaxe, (alm1.lmaxe, alm2.lmaxe)
        assert alm1.lmaxb == alm2.lmaxb, (alm1.lmaxb, alm2.lmaxb)

        ret = np.sum(hp.alm2cl(alm1.tlm, alm2.tlm) * (2. * np.arange(0, alm1.lmaxt + 1) + 1))
        ret += np.sum(hp.alm2cl(alm1.elm, alm2.elm) * (2. * np.arange(0, alm1.lmaxe + 1) + 1))
        ret += np.sum(hp.alm2cl(alm1.blm, alm2.blm) * (2. * np.arange(0, alm1.lmaxb + 1) + 1))
        return ret


class ForwardOperator:
    def __init__(self, s_inv_filt, n_inv_filt):
        lmax = len(n_inv_filt.tf1d_t) - 1
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
    def __init__(self, s_cls, n_inv_filt, nl_res=None, te_only=True):
        # diagonal pre-conditioner for conjugate-gradient solver
        # skip n_ell component (fine if white noise in ninv has most of the non-signal)
        lmax = len(n_inv_filt.tf1d_t) - 1
        if nl_res is None: nl_res = {key: np.zeros(lmax + 1) for key in s_cls}

        s_inv_filt = SkyInverseFilter(s_cls, nl_res, lmax, n_inv_filt.tf1d_t, n_inv_filt.tf1d_p, n_inv_filt.tf2d_t,
                                      n_inv_filt.tf2d_p)
        assert ((s_inv_filt.lmax + 1) >= len(n_inv_filt.tf1d_t))

        # Compute noise level
        ninv_ftl, ninv_fel, ninv_fbl = n_inv_filt.get_ftebl()

        # Define diagonal 1/(signal+noise)
        flmat = s_inv_filt.slinv[0:lmax + 1, :, :]
        flmat[:, 0, 0] += ninv_ftl
        flmat[:, 1, 1] += ninv_fel
        flmat[:, 2, 2] += ninv_fbl

        # Probably offset because this is signal + noise res whereas it shoudl be signal + total res  
        flmat = np.linalg.pinv(flmat)
        self.flmat = flmat
        self.te_only = te_only  # s_inv_filt.te_only

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, alm):
        tmat = self.flmat
        if self.te_only:
            rtlm = hp.almxfl(alm.tlm, tmat[:, 0, 0]) + hp.almxfl(alm.elm, tmat[:, 0, 1])
            relm = hp.almxfl(alm.tlm, tmat[:, 1, 0]) + hp.almxfl(alm.elm, tmat[:, 1, 1])
            rblm = hp.almxfl(alm.blm, tmat[:, 2, 2])
        else:
            rtlm = hp.almxfl(alm.tlm, tmat[:, 0, 0]) + hp.almxfl(alm.elm, tmat[:, 0, 1]) + hp.almxfl(alm.blm,
                                                                                                     tmat[:, 0, 2])
            relm = hp.almxfl(alm.tlm, tmat[:, 1, 0]) + hp.almxfl(alm.elm, tmat[:, 1, 1]) + hp.almxfl(alm.blm,
                                                                                                     tmat[:, 1, 2])
            rblm = hp.almxfl(alm.tlm, tmat[:, 2, 0]) + hp.almxfl(alm.elm, tmat[:, 2, 1]) + hp.almxfl(alm.blm,
                                                                                                     tmat[:, 2, 2])
        return teblm([rtlm, relm, rblm])


class SkyInverseFilter:  # alm_filter_sinv:
    # def __init__(self, s_cls, lmax, n_cls=None, tf2d=None, tf2d_eb=None,
    #             b_transf=None, b_transf_eb=None, cmb2d=None):
    def __init__(self, s_cls, nl_res, lmax, tf1d_t, tf1d_p, tf2d_t, tf2d_p):

        print("INITIALIZING SkyInverseFilter")
        self.lmax = lmax
        self.tf2d_t = tf2d_t
        self.tf2d_p = tf2d_p
        self.s_cls = s_cls
        self.nl_res = nl_res

        cltt = s_cls.get('tt', np.zeros(lmax + 1))[:lmax + 1]
        clte = s_cls.get('te', np.zeros(lmax + 1))[:lmax + 1]
        clee = s_cls.get('ee', np.zeros(lmax + 1))[:lmax + 1]
        clbb = s_cls.get('bb', np.zeros(lmax + 1))[:lmax + 1]

        nltt = nl_res.get('tt', np.zeros(lmax + 1))[:lmax + 1] / tf1d_t ** 2
        nlee = nl_res.get('ee', np.zeros(lmax + 1))[:lmax + 1] / tf1d_p ** 2
        nlbb = nl_res.get('bb', np.zeros(lmax + 1))[:lmax + 1] / tf1d_p ** 2

        cltt_2d = hp_utils.cl2almformat(cltt)
        clte_2d = hp_utils.cl2almformat(clte)
        clee_2d = hp_utils.cl2almformat(clee)
        clbb_2d = hp_utils.cl2almformat(clbb)

        nltt_2d = hp_utils.cl2almformat(nltt)
        nlee_2d = hp_utils.cl2almformat(nlee)
        nlbb_2d = hp_utils.cl2almformat(nlbb)

        # FIll in matrix with elements Clxx + |nlm_x|^2/TF^2
        slmat_alm = np.zeros((len(tf2d_t), 3, 3))
        slmat_alm[:, 0, 0] = cltt_2d + nltt_2d
        slmat_alm[:, 0, 1] = clte_2d
        slmat_alm[:, 1, 0] = clte_2d
        slmat_alm[:, 1, 1] = clee_2d + nlee_2d
        slmat_alm[:, 2, 2] = clbb_2d + nlbb_2d

        # slmat_alm[slmat_alm==0]=1e30
        self.slinv_alm = np.linalg.pinv(slmat_alm)

        print('--------------------')
        print(np.max(self.slinv_alm))
        # import pdb;pdb.set_trace()
        '''
        #import pdb; pdb.set_trace()
        if self.n_cls is not None:
            print("WE DO HAVE n_CLS")
            self.ncls1ds = self.ncls1d = {'tt': np.sqrt(hp.alm2cl(n_cls['tt'])),
                                          'ee': np.sqrt(hp.alm2cl(n_cls['ee'])),
                                          'bb': np.sqrt(hp.alm2cl(n_cls['bb']))}
        else:
            #sys.exit('bad path -- SkyInverseFilter/if n_cls is not None/else')
            self.ncls1ds = {a: np.zeros(lmax+1) for a in s_cls.keys()}
            #self.ncls1d = n_cls
        '''
        '''
        import pdb; pdb.set_trace()
        if b_transf is None:
            assert tf2d is None, "if tf2d is not None, nor should b_transf"
            b_transf = np.ones(lmax+1)
            sys.exit('bad path -- SkyInverseFilter/if b_transf is None:')

        if b_transf_eb is None: b_transf_eb = b_transf
        '''

        # always initialize diagonal (no l,m difference) S-inv; for PreOperatorDiag
        slmat = np.zeros((lmax + 1, 3, 3))  # matrix of TEB correlations at each l.
        slmat[:, 0, 0] = cltt + nltt
        slmat[:, 0, 1] = clte
        slmat[:, 1, 0] = slmat[:, 0, 1]
        slmat[:, 0, 2] = clte * 0
        slmat[:, 2, 0] = slmat[:, 0, 2]
        slmat[:, 1, 1] = clee + nlee
        slmat[:, 1, 2] = clee * 0
        slmat[:, 2, 1] = slmat[:, 1, 2]
        slmat[:, 2, 2] = clbb + nlbb
        slinv = np.linalg.pinv(slmat)

        self.slinv = slinv

        self.te_only = True

        if np.any(slmat[:, 0, 2]) or np.any(slmat[:, 1, 2]):
            self.te_only = False
            print('Found non-zero TB/EB')
            print('Setting TE flag to False')
        else:
            print('Setting TE Flag to True')

        print("DONE SkyInverseFilter--------------------")

    def calc(self, alm):
        # tmat = self.slinv

        if self.nl_res is not None and self.tf2d_t is not None:
            assert self.te_only, "only has te_only case"
            # This is the case where EB=BE=TB=BT==0
            rtlm = alm.tlm * self.slinv_alm[:, 0, 0] + alm.elm * self.slinv_alm[:, 0, 1]
            relm = alm.tlm * self.slinv_alm[:, 1, 0] + alm.elm * self.slinv_alm[:, 1, 1]
            rblm = alm.blm * self.slinv_alm[:, 2, 2]
        else:
            sys.exit('bad path -- SkyInverseFilter/calc(self, alm):')

        return teblm([rtlm, relm, rblm])

    def hashdict(self):
        return {'slinv': clhash(self.slinv.flatten())}


class NoiseInverseFilter:
    def __init__(self, n_inv, tf1d_t, tf1d_p, tf2d_t, tf2d_p):

        self.n_inv = []
        for i, tn in enumerate(n_inv):
            if isinstance(tn, list):
                n_inv_prod = hp_utils.read_map(tn[0][:])
                if len(tn) > 1:
                    for n in tn[1:]:
                        n_inv_prod = n_inv_prod * hp_utils.read_map(n[:])
                self.n_inv.append(n_inv_prod)
                # assert (np.std(self.n_inv[i][np.where(self.n_inv[i][:] != 0.0)]) / np.average(
                #    self.n_inv[i][np.where(self.n_inv[i][:] != 0.0)]) < 1.e-7)
            else:
                self.n_inv.append(hp_utils.read_map(n_inv[i]))

        n_inv = self.n_inv
        npix = len(n_inv[0])
        nside = hp.npix2nside(npix)
        for n in n_inv[1:]:
            assert (len(n) == npix)

        self.npix = npix
        self.nside = nside

        self.tf1d_t = tf1d_t
        self.tf1d_p = tf1d_p if tf1d_p is not None else self.tf1d_t
        assert len(self.tf1d_t) == len(self.tf1d_p)
        # self.b_transf = (self.b_transf_t + self.b_transf_e + self.b_transf_t) / 3.

        self.tf2d_t = tf2d_t
        self.tf2d_p = tf2d_p if tf2d_p is not None else tf2d_t

    def get_ftebl(self):
        if len(self.n_inv) == 2:  # TT, 1/2(QQ+UU)
            n_inv_cl_t = np.sum(self.n_inv[0]) / (4.0 * np.pi) * self.tf1d_t ** 2
            n_inv_cl_e = np.sum(self.n_inv[1]) / (4.0 * np.pi) * self.tf1d_p ** 2
            n_inv_cl_b = np.sum(self.n_inv[1]) / (4.0 * np.pi) * self.tf1d_p ** 2
            return n_inv_cl_t, n_inv_cl_e, n_inv_cl_b

        elif len(self.n_inv) == 4:  # TT, QQ, QU, UU
            sys.exit('bad path -- NoiseInverseFilter/get_ftebl(self)/if len(self.n_inv) == 2:')
            n_inv_cl_t = np.sum(self.n_inv[0]) / (4.0 * np.pi) * self.tf1d_t ** 2
            n_inv_cl_e = np.sum(0.5 * (self.n_inv[1] + self.n_inv[3])) / (4.0 * np.pi) * self.tf1d_p ** 2
            n_inv_cl_b = np.sum(0.5 * (self.n_inv[1] + self.n_inv[3])) / (4.0 * np.pi) * self.tf1d_p ** 2
            return n_inv_cl_t, n_inv_cl_e, n_inv_cl_b
        else:
            sys.exit('bad path -- NoiseInverseFilter/get_ftebl(self)/else:')

    def hashdict(self):
        sys.exit('bad path -- NoiseInverseFilter/hashdict(self):')
        return {'n_inv': [clhash(n) for n in self.n_inv],
                'b_transf': clhash(self.b_transf),
                }

    def apply_alm(self, alm):
        # applies Y^T N^{-1} Y
        lmax = alm.lmax

        if self.tf2d_t is None:
            sys.exit('bad path -- NoiseInverseFilter/apply_alm(self, alm):')
            # hp.almxfl(alm.tlm, self.b_transf_t, inplace=True)
            # hp.almxfl(alm.elm, self.b_transf_e, inplace=True)
            # hp.almxfl(alm.blm, self.b_transf_b, inplace=True)
        else:
            alm.tlm *= self.tf2d_t
            alm.elm *= self.tf2d_p
            alm.blm *= self.tf2d_p

        tmap, qmap, umap = hp.alm2map((alm.tlm, alm.elm, alm.blm), self.nside)
        # qmap, umap = hp.alm2map_spin((alm.elm, alm.blm), self.nside, 2)

        self.apply_map([tmap, qmap, umap])

        ttlm, telm, tblm = hp.map2alm([tmap, qmap, umap], lmax=lmax, iter=0, pol=True)
        alm.tlm[:] = ttlm
        alm.elm[:] = telm
        alm.blm[:] = tblm

        alm.tlm[:] *= (self.npix / (4. * np.pi))
        alm.elm[:] *= (self.npix / (4. * np.pi))
        alm.blm[:] *= (self.npix / (4. * np.pi))

        if self.tf2d_t is None:
            sys.exit('bad path -- NoiseInverseFilter/if self.tf2d_t is None:')
            # hp.almxfl(alm.tlm, self.b_transf_t, inplace=True)
            # hp.almxfl(alm.elm, self.b_transf_e, inplace=True)
            # hp.almxfl(alm.blm, self.b_transf_b, inplace=True)
        else:
            alm.tlm *= self.tf2d_t
            alm.elm *= self.tf2d_p
            alm.blm *= self.tf2d_p

    def apply_map(self, amap):
        [tmap, qmap, umap] = amap

        # applies N^{-1}
        if len(self.n_inv) == 2:  # TT, QQ=UU
            tmap *= self.n_inv[0]
            qmap *= self.n_inv[1]
            umap *= self.n_inv[1]

        elif len(self.n_inv) == 4:  # TT, QQ, QU, UU
            sys.exit('bad path -- NoiseInverseFilter/apply_map(self, amap)/elif len(self.n_inv) == 4:')
            # qmap_copy = qmap.copy()

            # tmap *= self.n_inv[0]
            # qmap *= self.n_inv[1]
            # qmap += self.n_inv[2] * umap

            # umap *= self.n_inv[3]
            # umap += self.n_inv[2] * qmap_copy

            # del qmap_copy
        else:
            assert 0
