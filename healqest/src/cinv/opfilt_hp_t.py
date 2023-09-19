"""
Similar to opfilt_teb.py for flatsky, this is for T-only and for healpix maps.

Modified from and built on plancklens/qcinv/opfilt_tt.py
"""
import os,sys
import hashlib
import numpy  as np
import healpy as hp

from healpy import alm2map, map2alm

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
import hp_utils
import cinv_utils
import pdb


def calc_prep(m, s_inv_filt, n_inv_filt):
    tmap = np.copy(m)
    #pdb.set_trace()
    n_inv_filt.apply_map(tmap)
    alm = map2alm(tmap, lmax=len(n_inv_filt.b_transf) - 1, iter=0)
    if n_inv_filt.tf2d is None:
        hp.almxfl(alm, n_inv_filt.b_transf * (len(m) / (4. * np.pi)), inplace=True)
    else:
        alm *= n_inv_filt.tf2d * (len(m) / (4. * np.pi))
    return alm


def calc_fini(alm, s_inv_filt, n_inv_filt):
    """ This final operation turns the Wiener-filtered CMB cg-solution to the inverse-variance filtered CMB.  """
    s_inv_filt.calc(alm, inplace=True)

class DotOperator:
    """ Scalar product definition for cg-inversion """
    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        lmax1 = hp.Alm.getlmax(alm1.size)
        assert lmax1 == hp.Alm.getlmax(alm2.size)
        return np.sum(hp.alm2cl(alm1, alms2=alm2) * (2. * np.arange(0, lmax1 + 1) + 1))

class ForwardOperator:
    """Conjugate-gradient inversion forward operation definition. """
    def __init__(self, s_inv_filt, n_inv_filt):
        self.s_inv_filt = s_inv_filt
        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return {'s_inv_filt': self.s_inv_filt.hashdict(),
                'n_inv_filt': self.n_inv_filt.hashdict()}

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        if np.all(talm == 0):  # do nothing if zero
            return talm
        nlm = np.copy(talm)
        self.n_inv_filt.apply_alm(nlm)
        slm = self.s_inv_filt.calc(talm)
        return nlm + slm

class PreOperatorDiag:
    def __init__(self, s_cls, n_inv_filt, n_cls=None):
        """Harmonic space diagonal pre-conditioner operation. """
        """returns  1/(1/S + 1/N)"""
        cltt = s_cls['tt']
        assert len(cltt) >= len(n_inv_filt.b_transf)
        n_inv_cl = np.sum(n_inv_filt.n_inv) / (4.0 * np.pi)
        lmax = len(n_inv_filt.b_transf) - 1
        assert lmax <= (len(cltt) - 1)
        if n_cls is None: n_cls = {key: np.zeros(lmax+1) for key in s_cls}

        filt = cinv_utils.cli(cltt[:lmax + 1] + n_cls['tt']*cinv_utils.cli(n_inv_filt.b_transf[:lmax + 1] ** 2))
        filt += n_inv_cl * n_inv_filt.b_transf[:lmax + 1] ** 2
        self.filt = cinv_utils.cli(filt)

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return hp.almxfl(talm, self.filt)

class SkyInverseFilter: #alm_filter_sinv_nocorr:
    """Class allowing spectrum-formed covariances
       from signal and also 1/ell noise. 
       For non-TT-only cases, does not include TE correlation
    """
    def __init__(self, s_cls, lmax, n_cls=None, tf2d=None, b_transf=None):

        self.n_cls = n_cls

        if n_cls is not None and tf2d is not None:
            #the only case that needs to expand into alms
            assert 'tt' in n_cls.keys()
            assert len(n_cls['tt']) == len(tf2d), "n_cls must be 2d (and TF+beam-ed)"

            cltt = s_cls['tt'][:lmax+1]
            cltt_2d = hp_utils.cl2almformat(cltt)

            #1/(cltt_2d + nltt_2d/tf2d**2) recovers 1d behavior if nltt_2d==0. 
            #the following two implementation are identical if tf2d has no zeros
            #if tf2d has zeros, this first one ensures the solution doesn't
            #depend on those modes
            self.slinv = tf2d*tf2d * cinv_utils.cli(cltt_2d *tf2d*tf2d + n_cls['tt'])
            #self.slinv = utils.cli(cltt_2d + n_cls['tt'] * utils.cli(tf2d*tf2d) )

        else:
            sys.exit('bad path')
            cltt = s_cls['tt'][:lmax+1]
            if n_cls is None: n_cls = {key: np.zeros(lmax+1) for key in s_cls}
            if b_transf is None:
                assert tf2d is None, "if tf2d is not None, nor should b_transf"
                b_transf = np.ones(lmax+1)
            pdb.set_trace()
            self.slinv = cinv_utils.cli(cltt + n_cls['tt']*cinv_utils.cli(b_transf*b_transf))

        self.lmax = lmax
        self.s_cls = s_cls
        self.tf2d  = tf2d
        if self.n_cls is not None and tf2d is not None:
            self.ncls1d = {'tt': np.sqrt(hp.alm2cl(n_cls['tt']))}
        else: #self.n_cls is None or self.n_cls is 1d
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

class NoiseInverseFilter: #alm_filter_ninv(object):
    """Missing doc. """
    def __init__(self, n_inv, b_transf,
                 tf2d = None,
                 nlev_ftl=None):
                 #marge_monopole=False, marge_dipole=False, marge_uptolmin=-1, marge_maps=(), nlev_ftl=None):
        if isinstance(n_inv, list):
            n_inv_prod = hp_utils.read_map(n_inv[0])
            if len(n_inv) > 1:
                for n in n_inv[1:]:
                    n_inv_prod = n_inv_prod * hp_utils.read_map(n)
            n_inv = n_inv_prod
        else:
            n_inv = hp_utils.read_map(n_inv)
        print("opfilt_tt: inverse noise map std dev / av = %.3e" % (
                    np.std(n_inv[np.where(n_inv != 0.0)]) / np.average(n_inv[np.where(n_inv != 0.0)])))

        self.n_inv = n_inv
        self.b_transf = b_transf
        self.tf2d = tf2d
        self.npix = len(self.n_inv)
        self.nside = hp.npix2nside(self.npix)

        if nlev_ftl is None:
            nlev_ftl =  10800. / np.sqrt(np.sum(self.n_inv) / (4.0 * np.pi)) / np.pi
        self.nlev_ftl = nlev_ftl
        print("ninv_ftl: using %.2f uK-amin noise Cl"%self.nlev_ftl)

    def hashdict(self):
        return {'n_inv': cinv_utils.clhash(self.n_inv),
                'b_transf': cinv_utils.clhash(self.b_transf)}

    def apply_alm(self, alm):
        """Missing doc. """
        npix = len(self.n_inv)
        if self.tf2d is None:
            hp.almxfl(alm, self.b_transf, inplace=True)
        else:
            alm *= self.tf2d
        tmap = alm2map(alm, hp.npix2nside(npix))
        self.apply_map(tmap)
        alm[:] = map2alm(tmap, lmax=hp.Alm.getlmax(alm.size),use_pixel_weights=True)
        if self.tf2d is None:
            hp.almxfl(alm, self.b_transf  *  (npix / (4. * np.pi)), inplace=True)
        else:
            alm *= self.tf2d * (npix / (4. * np.pi))

    def apply_map(self, tmap):
        """Missing doc. """
        tmap *= self.n_inv
