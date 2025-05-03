"""
Similar to opfilt_teb.py for flatsky, this is for T-only and for healpix maps.

Modified from and built on plancklens/qcinv/opfilt_tt.py
"""
import sys
import numpy as np
import healpy as hp

from healpy import alm2map, map2alm
from . import hp_utils, cinv_utils
import pdb


def calc_prep(m, s_inv_filt, n_inv_filt):
    tmap = np.copy(m)
    n_inv_filt.apply_map(tmap)
    alm = map2alm(tmap, lmax=len(n_inv_filt.b_transf) - 1, iter=0)
    pixarea = hp.nside2pixarea(hp.get_nside(m))
    if n_inv_filt.tf2d is None:
        hp.almxfl(alm, n_inv_filt.b_transf, inplace=True)
    else:
        alm *= n_inv_filt.tf2d
    return alm/pixarea


def calc_fini(alm, s_inv_filt, n_inv_filt):
    """This final operation turns the Wiener-filtered CMB cg-solution to the inverse-variance filtered CMB."""
    s_inv_filt.calc(alm, inplace=True)


class DotOperator:
    """Scalar product definition for cg-inversion"""

    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        lmax1 = hp.Alm.getlmax(alm1.size)
        assert lmax1 == hp.Alm.getlmax(alm2.size)
        return np.sum(hp.alm2cl(alm1, alms2=alm2) * (2.0 * np.arange(0, lmax1 + 1) + 1))


class ForwardOperator:
    """Conjugate-gradient inversion forward operation definition."""

    def __init__(self, s_inv_filt, n_inv_filt):
        self.s_inv_filt = s_inv_filt
        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return {
            "s_inv_filt": self.s_inv_filt.hashdict(),
            "n_inv_filt": self.n_inv_filt.hashdict(),
        }

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        # if np.all(talm == 0):  # do nothing if zero
        #    sys.exit('np.all(talm == 0)')
        #    return talm
        nlm = np.copy(talm)
        self.n_inv_filt.apply_alm(nlm)
        slm = self.s_inv_filt.calc(talm)
        return nlm + slm


class PreOperatorDiag:
    def __init__(self, s_cls, n_inv_filt, n_cls=None):
        """Harmonic space diagonal pre-conditioner operation."""
        """returns  1/(1/S + 1/N)"""
        cltt = s_cls["tt"]
        assert len(cltt) >= len(n_inv_filt.b_transf)

        # n_inv_cl = np.sum(n_inv_filt.n_inv) / (4.0 * np.pi)
        ninv_ftl = n_inv_filt.get_ftl()

        lmax = len(n_inv_filt.b_transf) - 1
        assert lmax <= (len(cltt) - 1)
        if n_cls is None:
            n_cls = {key: np.zeros(lmax + 1) for key in s_cls}

        filt = cinv_utils.cli(
            cltt[: lmax + 1]
            + n_cls["tt"] * cinv_utils.cli(n_inv_filt.b_transf[: lmax + 1] ** 2)
        )
        filt += ninv_ftl  # * n_inv_filt.b_transf[:lmax + 1] ** 2
        self.filt = cinv_utils.cli(filt)

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return hp.almxfl(talm, self.filt)


class SkyInverseFilter:  # alm_filter_sinv_nocorr:
    """Class allowing spectrum-formed covariances
    from signal and also 1/ell noise.
    For non-TT-only cases, does not include TE correlation
    """

    def __init__(self, s_cls, lmax, n_cls=None, tf2d=None, b_transf=None, slinv=None):
        self.n_cls = n_cls

        cltt = s_cls["tt"][:lmax + 1]
        if n_cls is not None and tf2d is not None:
            # the only case that needs to expand into alms
            assert "tt" in n_cls.keys()
            assert len(n_cls["tt"]) == len(tf2d), "n_cls must be 2d (and TF+beam-ed)"
            assert n_cls['tt'] == tf2d.shape
            cltt_2d = hp_utils.cl2almformat(cltt)  # dltt

            # 1/(cltt_2d + nltt_2d/tf2d**2) recovers 1d behavior if nltt_2d==0.
            # the following two implementation are identical if tf2d has no zeros
            # if tf2d has zeros, this first one ensures the solution doesn't
            # depend on those modes
            self.slinv = (tf2d * tf2d * cinv_utils.cli(cltt_2d * tf2d * tf2d + n_cls["tt"]))
            # self.slinv = utils.cli(cltt_2d + n_cls['tt'] * utils.cli(tf2d*tf2d) )
        else:
            # raise ValueError("NOT: n_cls is not None and tf2d is not None:")
            if n_cls is None:
                n_cls = {key: np.zeros(lmax + 1) for key in s_cls}
            if b_transf is None:
                assert tf2d is None, "if tf2d is not None, nor should b_transf"
                b_transf = np.ones(lmax + 1)
            self.slinv = cinv_utils.cli(cltt + n_cls["tt"] * cinv_utils.cli(b_transf * b_transf))

        self.lmax = lmax
        self.s_cls = s_cls
        self.tf2d = tf2d

        if self.n_cls is not None and tf2d is not None:
            self.ncls1d = {"tt": np.sqrt(hp.alm2cl(n_cls["tt"]))}

        else:  # self.n_cls is None or self.n_cls is 1d
            self.ncls1d = self.n_cls

    def calc(self, alm, inplace=False):
        if self.n_cls is not None and self.tf2d is not None:
            if inplace:
                alm *= self.slinv
            else:
                return alm * self.slinv
        else:
            sys.exit("bad path")
            if inplace:
                hp.almxfl(alm, self.slinv, inplace=inplace)
            else:
                return hp.almxfl(alm, self.slinv, inplace=inplace)

    def hashdict(self):
        return {"slinv": cinv_utils.clhash(self.slinv)}


"""
class SkyInverseFilter: #alm_filter_sinv_nocorr:
    def __init__(self, s_cls, lmax, n_cls=None, tf2d=None, b_transf=None, obsstack=None):
        
        s_cls    = dl
        lmax     = lmax
        n_cls    = nl_dl
        tf2d     = tf2d_dl
        b_transf = ones* ell*(ell+1)/2/np.pi
        

        self.n_cls = n_cls # dict of |nlm|^2 * l*(l+1)/2/pi

        if n_cls is not None and tf2d is not None:
            #the only case that needs to expand into alms
            assert 'tt' in n_cls.keys()
            assert len(n_cls['tt']) == len(tf2d), "n_cls must be 2d (and TF+beam-ed)"

            if obsstack is None:
                sys.exit('bad path')
                cltt    = s_cls['tt'][:lmax+1]
                cltt_2d = hp_utils.cl2almformat(cltt) # dltt

                #1/(cltt_2d + nltt_2d/tf2d**2) recovers 1d behavior if nltt_2d==0. 
                #the following two implementation are identical if tf2d has no zeros
                #if tf2d has zeros, this first one ensures the solution doesn't
                #depend on those modes
                self.slinv = tf2d*tf2d * cinv_utils.cli(cltt_2d *tf2d*tf2d + n_cls['tt'])
                #self.slinv = utils.cli(cltt_2d + n_cls['tt'] * utils.cli(tf2d*tf2d) )
            else:
                u = obsstack['tt']*tf2d*tf2d
                u[u<1e-6]=np.inf
                self.slinv = tf2d*tf2d * cinv_utils.cli(u)
                #import pdb; pdb.set_trace()
        else:
            sys.exit('bad path')
            cltt = s_cls['tt'][:lmax+1]
            if n_cls is None: n_cls = {key: np.zeros(lmax+1) for key in s_cls}
            if b_transf is None:
                assert tf2d is None, "if tf2d is not None, nor should b_transf"
                b_transf = np.ones(lmax+1)
            pdb.set_trace()
            self.slinv = cinv_utils.cli(cltt + n_cls['tt']*cinv_utils.cli(b_transf*b_transf))

        self.lmax  = lmax
        self.s_cls = s_cls
        self.tf2d  = tf2d

        if self.n_cls is not None and tf2d is not None:
            # Compute 1d noise
            self.ncls1d = {'tt': np.sqrt(hp.alm2cl(n_cls['tt']))}

        else: #self.n_cls is None or self.n_cls is 1d
            sys.exit('bad path')
            self.ncls1d = self.n_cls

    def calc(self, alm, inplace=False):
        if self.n_cls is not None and self.tf2d is not None:
            if inplace:
                alm *= self.slinv
            else:
                return alm*self.slinv
        else:
            if inplace:
                hp.almxfl(alm, self.slinv, inplace=inplace)
            else:
                return hp.almxfl(alm, self.slinv, inplace=inplace)


    def hashdict(self):
        return {'slinv': cinv_utils.clhash(self.slinv)}
"""


class NoiseInverseFilter:  # alm_filter_ninv(object):
    """Missing doc."""

    def __init__(self, n_inv, b_transf, tf2d=None, nlev_ftl=None):
        # marge_monopole=False, marge_dipole=False, marge_uptolmin=-1, marge_maps=(), nlev_ftl=None):
        if isinstance(n_inv, list):
            print("isinstance(n_inv, list)")
            n_inv_prod = hp_utils.read_map(n_inv[0])
            if len(n_inv) > 1:
                for n in n_inv[1:]:
                    n_inv_prod = n_inv_prod * hp_utils.read_map(n)
            n_inv = n_inv_prod
        else:
            print("n_inv is NOT A LIST")
            n_inv = hp_utils.read_map(n_inv)

        ninv_std = np.std(n_inv[np.where(n_inv != 0.0)])
        ninv_avg = np.average(n_inv[np.where(n_inv != 0.0)])
        print(f"opfilt_t: inverse noise map std dev / av = {ninv_std/ninv_avg:.3e}")

        self.n_inv = n_inv
        self.b_transf = b_transf
        self.tf2d = tf2d
        self.npix = len(self.n_inv)
        self.nside = hp.npix2nside(self.npix)

        self.nlev_ftl = nlev_ftl
        self._n_inv = n_inv  # could be paths or list of paths
        self._load_ninv()

    def hashdict(self):
        return {
            "n_inv": cinv_utils.clhash(self.n_inv),
            "b_transf": cinv_utils.clhash(self.b_transf),
        }

    '''
    def apply_alm(self, alm):
        """Missing doc. """
        npix = len(self.n_inv) #npix of 2048
        if self.tf2d is None:
            sys.exit('bad paths sss')
            hp.almxfl(alm, self.b_transf, inplace=True)
        else:
            alm *= self.tf2d # alm*TF*bl

        tmap = alm2map(alm, hp.npix2nside(npix))
        
        self.apply_map(tmap)
        alm[:] = map2alm(tmap, lmax=hp.Alm.getlmax(alm.size),use_pixel_weights=True)
        
        if self.tf2d is None:
            sys.exit('bad path ddddd')
            hp.almxfl(alm, self.b_transf  *  (npix / (4. * np.pi)), inplace=True)
        else:
            alm *= self.tf2d * (npix / (4. * np.pi)) #sr per pixel

    def apply_map(self, tmap):
        """Missing doc. """
        tmap *= self.n_inv
    '''

    def _load_ninv(self):
        print("==_load_ninv==")
        if self.n_inv is None:
            self.n_inv = []
            for i, tn in enumerate(self._n_inv):
                if isinstance(tn, list):
                    sys.exit("BAD PATH -- multiple n_inv")
                    n_inv_prod = hp_utils.read_map(tn[0])
                    if len(tn) > 1:
                        for n in tn[1:]:
                            n_inv_prod = n_inv_prod * hp_utils.read_map(n)
                    self.n_inv.append(n_inv_prod)
                else:
                    print("Loading single n_inv")
                    self.n_inv.append(hp_utils.read_map(self._n_inv[i]))
            assert len(self.n_inv) in [1, 3], len(self.n_inv)
            self.nside = hp.npix2nside(len(self.n_inv[0]))
        else:
            print("self.n_inv is not None")

    def apply_alm(self, alm):
        """B^dagger N^{-1} B"""
        print("apply_alm")
        self._load_ninv()
        lmax = hp.Alm.getlmax(alm.shape[0])

        if self.tf2d is None:
            # hp.almxfl(alm.elm, self.b_transf_e, inplace=True)
            # hp.almxfl(alm.blm, self.b_transf_b, inplace=True)
            sys.exit("no tf2d supplied")
        else:
            alm *= self.tf2d

        tmap = alm2map(alm, self.nside)
        # qmap, umap = alm2map_spin((alm.elm, alm.blm), self.nside, 2, lmax)

        self.apply_map(tmap)  # applies N^{-1}
        npix = len(tmap)

        ttlm = map2alm(tmap, lmax=lmax, iter=0)
        alm[:] = ttlm

        if self.tf2d is None:
            # hp.almxfl(alm.elm, self.b_transf_e * (npix / (4. * np.pi)), inplace=True)
            # hp.almxfl(alm.blm, self.b_transf_b * (npix / (4. * np.pi)), inplace=True)
            sys.exit("no tf2d supplied")
        else:
            alm *= self.tf2d * (npix / (4.0 * np.pi))
            # alm.blm *= self.tf2d * (npix / (4. * np.pi))

    def apply_map(self, amap):
        print("apply_map")
        self._load_ninv()
        tmap = amap
        # print( self.n_inv[0])
        # if len(self.n_inv) == 1:  # TT, QQ=UU
        #    print('only detected one n_inv map')
        tmap *= self.n_inv
        # else:
        #    assert 0

    def get_ftl(self):
        print("==get_febl==")
        if self.nlev_ftl is None:
            self.nlev_ftl = self._calc_ftl()
        n_inv_cl_t = self.b_transf**2 / (self.nlev_ftl / 180.0 / 60.0 * np.pi) ** 2
        return n_inv_cl_t

    def _calc_ftl(self):
        print("==_calc_febl==")
        self._load_ninv()
        # import pdb;pdb.set_trace()
        # if len(self.n_inv) == 1:
        nlev_ftl = 10800.0 / np.sqrt(np.sum(self.n_inv) / (4.0 * np.pi)) / np.pi
        # elif len(self.n_inv) == 3:
        #    nlev_ftl = 10800. / np.sqrt(np.sum(0.5 * (self.n_inv[0] + self.n_inv[2])) / (4.0 * np.pi)) / np.pi
        # else:
        #    assert 0
        print("ninv_febl: using %.2f uK-amin noise Cl" % nlev_ftl)
        return nlev_ftl
