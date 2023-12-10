"""
Classes holding C-inv related object for healpix maps
Copied from what is implemented by Kimmy in spt3g_software
https://github.com/SouthPoleTelescope/spt3g_software/blob/curvlens/lensing/python/cinv_hp.py
but with additional cleaning/formatting and commenting.
"""

import os,sys
import healpy as hp
import numpy  as np
import pickle as pk
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
import utils, hp_utils, cinv_utils
import opfilt_hp_p, opfilt_hp_t, cd_solve, cd_monitors
# disable mpi

class cinv(object):
    def __init__(self, lib_dir, lmax, eps_min, use_mpi=False):
        self.lib_dir = lib_dir # Output directory
        self.lmax    = lmax    # Lmax to use for filtering
        self.eps_min = eps_min # Tolerance

        if use_mpi==True:
            from cinv_utils import mpi

    def get_tal(self, a, lmax=None):
        if lmax is None: lmax = self.lmax
        assert a.lower() in ['t', 'e', 'b'], a
        ret = np.loadtxt(os.path.join(self.lib_dir, "tal.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax +1 ]

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))

    def get_ftl(self, lmax=None):
        if lmax is None: lmax = self.lmax
        ret = np.loadtxt(os.path.join(self.lib_dir, "ftl.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax + 1]

    def get_fel(self, lmax=None):
        if lmax is None: lmax = self.lmax
        ret = np.loadtxt(os.path.join(self.lib_dir, "fel.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax + 1]

    def get_fbl(self, lmax=None):
        if lmax is None: lmax = self.lmax
        ret = np.loadtxt(os.path.join(self.lib_dir, "fbl.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax + 1]

    def solve(self, soltn, tpn_map):
        '''
        soltn  : talm
        tqn_map: tmap
        '''
        finifunc = getattr(self.opfilt, 'calc_fini') 
        self.iter_tot = 0
        self.prev_eps = None
        dot_op  = self.opfilt.DotOperator()
        logger  = cd_monitors.logger_basic

        tpn_alm = self.opfilt.calc_prep(tpn_map, self.s_inv_filt, self.n_inv_filt)
        monitor = cd_monitors.MonitorBasic(dot_op, logger=logger, iter_max=np.inf, eps_min=self.eps_min) 
        fwd_op  = self.opfilt.ForwardOperator(self.s_inv_filt, self.n_inv_filt)
        pre_op  = self.opfilt.PreOperatorDiag(self.s_inv_filt.s_cls, 
                                              self.n_inv_filt,
                                              n_cls=self.s_inv_filt.ncls1d)
        
        cd_solve.cd_solve(soltn, b         = tpn_alm,
                                 fwd_op    = fwd_op, 
                                 pre_ops   = [pre_op],
                                 dot_op    = dot_op,
                                 criterion = monitor,
                                 tr        = cinv_utils.cd_solve.tr_cg, 
                                 cache     = cinv_utils.cd_solve.CacheMemory()       
                         )

        finifunc(soltn, self.s_inv_filt, self.n_inv_filt)

class cinv_t(cinv):
    r"""Temperature-only inverse-variance (or Wiener-)filtering instance.

        Args:
            lib_dir: mask and other things will be cached there
            lmax   : filtered alm's are reconstructed up to lmax
            nside  : healpy resolution of maps to filter
            cl     : fiducial CMB spectra used to filter the data (dict with 'tt' key)
            transf : CMB maps transfer function (array)
            ninv   : inverse pixel variance map. Must be a list of paths or of healpy maps with consistent nside.
            rescal_cl: isotropic rescaling of the map prior the cg inversion. This just makes the convergence criterium change a bit
            nl: 1/ell noise spec (dict with 'tt' key), with ninv-levels in uK^2 subtracted
            tf2d: transfer function in alms
        Note:

            The only difference of the original plancklens filter is the rescaling of the maps. In effect, the modes of :math'`D_\ell` rather than :math'`C_\ell` are reconstructed
            This changes nothing to the iterations, but gives the dot product testing for convergence more sensitvity to relevant scales

    """
    def __init__(self, lib_dir, lmax, nside, cl, transf, ninv, 
                 eps_min = 1.0e-5,
                 nl=None,
                 rescal_cl='default', tf2d=None):

        #np.save('/lcrc/project/SPT3G/users/ac.yomori/scratch/nlttaaa.npy',nl['tt'])
        #import pdb;pdb.set_trace()

        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (lib_dir, lmax, nside)
        assert isinstance(ninv, list)
        super(cinv_t, self).__init__(lib_dir, lmax, eps_min)

        # Convert Cls to Dls or not
        if rescal_cl in ['default', None]:
            #rescal_cl = np.sqrt(np.arange(lmax + 1, dtype=float)**0 * np.arange(1, lmax + 2, dtype=float) / 2. / np.pi)
            rescal_cl = np.ones(lmax+1)

        # mutltiply Cls and tf with l(l+1)/2/pi factor
        dl        = {k: rescal_cl ** 2 * cl[k][:lmax + 1] for k in cl.keys()}  # rescaled cls (Dls by default)
        transf_dl = transf[:lmax + 1] * cinv_utils.cli(rescal_cl)

        if tf2d is not None:
            tf2d_dl = hp.almxfl(tf2d, cinv_utils.cli(rescal_cl))
            self.rescaled_tf2d = tf2d_dl
        else:
            tf2d_dl = tf2d

        if nl is not None:
            print('nl was provided, applying rescal_cl')
            #print(rescal_cl)
            nl_dl = {key: hp.almxfl(nl[key], cinv_utils.cli(rescal_cl**2)) for key in nl}
            #np.save('/lcrc/project/SPT3G/users/ac.yomori/scratch/nlttaaa.npy',nl['tt'])
            #np.save('/lcrc/project/SPT3G/users/ac.yomori/scratch/nlttdl.npy',nl_dl['tt'])
            #import pdb; pdb.set_trace()
            #self.rescaled_nl   = nl_dl
        else:
            sys.exit('nl was NOT provided')
            nl_dl = nl

        self.nside = nside

        self.cl = cl
        self.dl = dl

        self.transf = transf[:lmax + 1]
        self.rescaled_transf =transf_dl
        self.rescal_cl = rescal_cl
      
        self.nl   = nl
        self.tf2d = tf2d
        self.ninv = ninv

        n_inv_filt = hp_utils.jit(opfilt_hp_t.NoiseInverseFilter, ninv, transf_dl, tf2d=tf2d_dl)
        s_inv_filt = hp_utils.jit(opfilt_hp_t.SkyInverseFilter  , dl, lmax, n_cls=nl_dl, tf2d=tf2d_dl, b_transf=transf_dl)
        self.n_inv_filt = n_inv_filt
        self.s_inv_filt = s_inv_filt
        self.opfilt = opfilt_hp_t

        #if mpi.rank == 0:
        Y=0
        if Y==0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            '''
            if not os.path.exists(os.path.join(lib_dir, "filt_hash.pk")):
                pk.dump(self.hashdict(), open(os.path.join(lib_dir, "filt_hash.pk"), 'wb'), protocol=2)

            if not os.path.exists(os.path.join(self.lib_dir, "ftl.dat")):
                np.savetxt(os.path.join(self.lib_dir, "ftl.dat"), self._calc_ftl())

            if not os.path.exists(os.path.join(self.lib_dir, "tal.dat")):
                np.savetxt(os.path.join(self.lib_dir, "tal.dat"),  self._calc_tal())

            '''
            if not os.path.exists(os.path.join(self.lib_dir, "fmask.fits.gz")):
                hp.write_map(os.path.join(self.lib_dir, "fmask.fits.gz"), self._calc_mask(),overwrite=True)
            
        #mpi.barrier()
        #cinv_utils.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict())

    def _ninv_hash(self):
        ret = []
        for ninv_comp in self.ninv:
            if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                ret.append(cinv_utils.clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
        return ret
    
    def _calc_ftl(self):
        '''
        Compute 1/(cltt+nltt/bl**2)
        '''
        ninv = self.n_inv_filt.n_inv
        npix = len(ninv[:])
        NlevT_uKamin = np.sqrt(4. * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0])) * 180. * 60. / np.pi
        print("cinv_t::noiseT_uk_arcmin = %.3f"%NlevT_uKamin)

        s_cls    = self.cl
        b_transf = self.transf

        ftl = cinv_utils.cli(s_cls['tt'][0:self.lmax + 1] + (NlevT_uKamin * np.pi / 180. / 60.) ** 2 / b_transf[0:self.lmax + 1] ** 2)
        ftl[0:2] = 0.0

        return ftl 

    def _calc_tal(self):
        '''
        Compute 1/tf (2d)
        '''
        return cinv_utils.cli(self.transf)

    def _calc_mask(self):
        '''
        Return binary mask 1 if pixval>0 and 0 otherwise
        '''
        ninv = self.n_inv_filt.n_inv
        assert hp.npix2nside(len(ninv)) == self.nside
        return np.where(ninv > 0, 1., 0.)

    def hashdict(self):
        return {'lmax': self.lmax,
                'nside': self.nside,
                'rescal_cl':cinv_utils.clhash(self.rescal_cl),
                'cltt': cinv_utils.clhash(self.cl['tt'][:self.lmax + 1]),
                'transf': cinv_utils.clhash(self.transf[:self.lmax + 1]),
                'ninv': self._ninv_hash(),
                }

    def apply_ivf(self, tmap, soltn=None):
        '''
        '''
        if soltn is None:
            talm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex_)
        else:
            talm = soltn.copy()

        self.solve(talm, tmap)
        hp.almxfl(talm, self.rescal_cl, inplace=True) #  Multiply in 2d by l(l+1)/(2*np.pi)
        return talm


class cinv_p(cinv):
    r"""Polarization-only inverse-variance (or Wiener-)filtering instance.

        Args:
            lib_dir : mask and other things will be cached there
            lmax    : filtered alm's are reconstructed up to lmax
            nside   : healpy resolution of maps to filter
            cl      : fiducial CMB spectra used to filter the data (dict with 'tt' key)
            transf  : CMB E-mode polarization transfer function (array)
            ninv    : inverse pixel variance maps. Must be a list of either 3 (QQ, QU, UU) or 1 (QQ = UU noise) elements.
                      These element are themselves list of paths or of healpy maps with consistent nside.
            transf_blm(optional): B-polarization transfer function (if different from E-mode one)

        Note:
            This implementation now supports template projection

    """
    def __init__(self, lib_dir, lmax, nside, cl, transf, ninv, 
                 eps_min = 1.0e-5,
                 nl=None, tf2d=None,
                 transf_blm=None):
        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (lib_dir, lmax, nside)
        super(cinv_p, self).__init__(lib_dir, lmax, eps_min)

        self.nside = nside
        self.cl = cl
        self.transf_e = transf
        self.transf_b = transf if transf_blm is None else transf_blm
        self.transf = transf if transf_blm is None else 0.5 * self.transf_e + 0.5 * self.transf_b
        self.tf2d = tf2d
        self.ninv = ninv
        self.nl = nl

        n_inv_filt = hp_utils.jit(opfilt_hp_p.NoiseInverseFilter, ninv, transf[0:lmax + 1], tf2d=self.tf2d,
                              b_transf_b=transf_blm) 
        s_inv_filt = hp_utils.jit(opfilt_hp_p.SkyInverseFilter, cl, lmax,
                              n_cls=nl, tf2d=tf2d, b_transf = self.transf)
        self.n_inv_filt = n_inv_filt
        self.s_inv_filt = s_inv_filt
        self.opfilt = opfilt_hp_p
        
        Y=0
        if Y==0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            '''
            if not os.path.exists(os.path.join(lib_dir, "filt_hash.pk")):
                pk.dump(self.hashdict(), open(os.path.join(lib_dir, "filt_hash.pk"), 'wb'), protocol=2)

            if not os.path.exists(os.path.join(self.lib_dir, "fbl.dat")):
                fel, fbl = self._calc_febl()
                np.savetxt(os.path.join(self.lib_dir, "fel.dat"), fel)
                np.savetxt(os.path.join(self.lib_dir, "fbl.dat"), fbl)

            if not os.path.exists(os.path.join(self.lib_dir, "tal.dat")):
                np.savetxt(os.path.join(self.lib_dir, "tal.dat"), self._calc_tal())
            '''
            if not os.path.exists(os.path.join(self.lib_dir,  "fmask.fits.gz")):
                hp.write_map(os.path.join(self.lib_dir,  "fmask.fits.gz"),  self._calc_mask())
            
        #mpi.barrier()
        #cinv_utils.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict())

    def hashdict(self):
        return {'lmax': self.lmax,
                'nside': self.nside,
                'clee': cinv_utils.clhash(self.cl.get('ee', np.array([0.]))),
                'cleb': cinv_utils.clhash(self.cl.get('eb', np.array([0.]))),
                'clbb': cinv_utils.clhash(self.cl.get('bb', np.array([0.]))),
                'transf':cinv_utils.clhash(self.transf),
                'ninv': self._ninv_hash()}

    def apply_ivf(self, tmap, soltn=None):
        if soltn is not None:
            assert len(soltn) == 2
            assert hp.Alm.getlmax(soltn[0].size) == self.lmax, (hp.Alm.getlmax(soltn[0].size), self.lmax)
            assert hp.Alm.getlmax(soltn[1].size) == self.lmax, (hp.Alm.getlmax(soltn[1].size), self.lmax)
            talm = hp_utils.eblm([soltn[0], soltn[1]])
        else:
            telm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
            tblm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
            talm = hp_utils.eblm([telm, tblm])

        assert len(tmap) == 2
        self.solve(talm, [tmap[0], tmap[1]])

        return talm.elm, talm.blm

    def _calc_febl(self):
        assert not 'eb' in self.cl.keys()

        if len(self.ninv) == 1:
            ninv = self.n_inv_filt.n_inv[0]
            npix = len(ninv)
            NlevP_uKamin = np.sqrt(
                4. * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0])) * 180. * 60. / np.pi
        else:
            assert len(self.ninv) == 3
            ninv = self.n_inv_filt.n_inv
            NlevP_uKamin= 0.5 * np.sqrt(
                4. * np.pi / len(ninv[0]) / np.sum(ninv[0]) * len(np.where(ninv[0] != 0.0)[0])) * 180. * 60. / np.pi
            NlevP_uKamin += 0.5 * np.sqrt(
                4. * np.pi / len(ninv[2]) / np.sum(ninv[2]) * len(np.where(ninv[2] != 0.0)[0])) * 180. * 60. / np.pi

        print("cinv_p::noiseP_uk_arcmin = %.3f"%NlevP_uKamin)

        s_cls = self.cl
        b_transf_e = self.n_inv_filt.b_transf_e
        b_transf_b = self.n_inv_filt.b_transf_b
        fel = cinv_utils.cli(s_cls['ee'][:self.lmax + 1] + (NlevP_uKamin * np.pi / 180. / 60.) ** 2 * cinv_utils.cli(b_transf_e[0:self.lmax + 1] ** 2))
        fbl = cinv_utils.cli(s_cls['bb'][:self.lmax + 1] + (NlevP_uKamin * np.pi / 180. / 60.) ** 2 * cinv_utils.cli(b_transf_b[0:self.lmax + 1] ** 2))

        fel[0:2] *= 0.0
        fbl[0:2] *= 0.0

        return fel, fbl

    def _calc_tal(self):
        return cinv_utils.cli(self.transf)

    def _calc_mask(self):
        mask = np.ones(hp.nside2npix(self.nside), dtype=float)
        for ninv in self.ninv:
            assert hp.npix2nside(len(ninv)) == self.nside
            mask *= (ninv > 0.)
        return mask

    def _ninv_hash(self):
        ret = []
        for ninv_comp in self.ninv[0]:
            if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                ret.append(cinv_utils.clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
        return [ret]

class library_sepTP(object):
    """
    Template class for CMB inverse-variance and Wiener-filtering library.
    This is suitable whenever the temperature and polarization maps are independently filtered.

    Args:
        lib_dir (str): directory where hashes and filtered maps will be cached.
        sim_lib      : simulation library instance. *sim_lib* must have *get_sim_tmap* and *get_sim_pmap* methods.
        cl_weights   : CMB spectra, used to compute the Wiener-filtered CMB from the inverse variance filtered maps.
        lfilt        : 1d lmin/lmax cuts to the output inverse-variance-filtered/Wiener-filtered maps (same for T,E,B)

    """
    def __init__(self, lib_dir, sim_lib, cl_weights, lfilt = None, soltn_lib = None, cache = True, add_noise=False):

        self.lib_dir   = lib_dir
        self.sim_lib   = sim_lib
        self.cl        = cl_weights
        self.lfilt     = lfilt
        self.add_noise = add_noise

        self.soltn_lib = soltn_lib
        self.cache     = cache
        fn_hash        = os.path.join(lib_dir, 'filt_hash.pk')

        Y=0
        if Y==0:
        #if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            #if not os.path.exists(fn_hash):
            #    pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        #mpi.barrier()
        #cinv_utils.hash_check(pk.load(open(fn_hash, 'rb')), self.hashdict())

    #def _apply_ivf_t(self, tmap, soltn=None):
    #    assert 0, 'override this'
    #def _apply_ivf_p(self, pmap, soltn=None):
    #    assert 0, 'override this'

    def get_sim_teblm(self, idx):
        """
        Returns an inverse-filtered T/E/B simulation.

        Args: idx    : simulation index
              Returns: inverse-filtered temperature healpy alm array
        """
        tfname = os.path.join(self.lib_dir, 'sim_%04d_tlm.fits'%idx if idx >= 0 else 'dat_tlm.fits')

        # Loading unfiltered alms
        if not os.path.exists(tfname):
            print("no idea what its supposed to do here")
            #tlm = self._apply_ivf_t(self.sim_lib.get_sim_tmap(idx), soltn=None if self.soltn_lib is None else self.soltn_lib.get_sim_tmliklm(idx))
            #if self.cache: hp.write_alm(tfname, tlm, overwrite=True)
        else:
            print('Loading file: %s'%tfname)
            tlm,elm,blm = hp.read_alm(tfname,hdu=[1,2,3])

        # Apply lmin/lmax cuts in 1d
        if self.lfilt is not None:
            hp.almxfl(tlm, self.lfilt, inplace=True)
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)

        return tlm,elm,blm


    def get_sim_tlm(self, idx):
        """
        Returns an inverse-filtered temperature simulation.
        
        Args: idx    : simulation index
              Returns: inverse-filtered temperature healpy alm array
        """
        tfname = os.path.join(self.lib_dir, 'sim_%04d_tlm.fits'%idx if idx >= 0 else 'dat_tlm.fits')
        if not os.path.exists(tfname):
            print("tlm file doesnt exit so creating one")
            tlm = self._apply_ivf_t(self.sim_lib.get_tmap(idx, add_noise=self.add_noise), soltn=None if self.soltn_lib is None else self.soltn_lib.get_sim_tmliklm(idx))
            #if self.cache: hp.write_alm(tfname, tlm, overwrite=True)
        else:
            print("Loading file: %s"%tfname)
            tlm = hp.read_alm(tfname)
        if self.lfilt is not None:
            hp.almxfl(tlm, self.lfilt, inplace=True)
        return tlm

    def get_sim_elm(self, idx):
        """Returns an inverse-filtered E-polarization simulation.
            Args:idx: simulation index
            Returns: inverse-filtered E-polarization healpy alm array
        """
        tfname = os.path.join(self.lib_dir, 'sim_%04d_elm.fits'%idx  if idx >= 0 else 'dat_elm.fits')
        if not os.path.exists(tfname):
            if self.soltn_lib is None:
                soltn = None
            else:
                soltn = np.array([self.soltn_lib.get_sim_emliklm(idx), self.soltn_lib.get_sim_bmliklm(idx)])

            elm, blm = self._apply_ivf_p(self.sim_lib.get_pmap(idx,add_noise=self.add_noise), soltn=soltn)

            if self.cache:
                hp.write_alm(tfname, elm, overwrite=True)
                hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_blm.fits'%idx if idx >= 0 else 'dat_blm.fits'), blm, overwrite=True)
        else:
            elm = hp.read_alm(tfname)
        if self.lfilt is not None:
            hp.almxfl(elm, self.lfilt, inplace=True)
        return elm

    def get_sim_eblm(self, idx):
        """Returns an inverse-filtered E-polarization simulation.
            Args:idx: simulation index
            Returns: inverse-filtered E-polarization healpy alm array
        """
        if self.soltn_lib is None:
            soltn = None
        else:
            soltn = np.array([self.soltn_lib.get_sim_emliklm(idx), self.soltn_lib.get_sim_bmliklm(idx)])

        elm, blm = self._apply_ivf_p(self.sim_lib.get_pmap(idx,add_noise=self.add_noise), soltn=soltn)


        '''
        tfname = os.path.join(self.lib_dir, 'sim_%04d_elm.fits'%idx  if idx >= 0 else 'dat_elm.fits')
        if not os.path.exists(tfname):
            print("elm file doesnt exit so creating one")

            if self.soltn_lib is None:
                soltn = None
            else:
                soltn = np.array([self.soltn_lib.get_sim_emliklm(idx), self.soltn_lib.get_sim_bmliklm(idx)])

            elm, blm = self._apply_ivf_p(self.sim_lib.get_pmap(idx,add_noise=self.add_noise), soltn=soltn)

            if self.cache:
                hp.write_alm(tfname, elm, overwrite=True)
                hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_blm.fits'%idx if idx >= 0 else 'dat_blm.fits'), blm, overwrite=True)
        else:
            elm = hp.read_alm(tfname)
        '''
        if self.lfilt is not None:
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)
        return elm,blm



    def get_sim_blm(self, idx):
        """Returns an inverse-filtered B-polarization simulation.
            Args: idx: simulation index
            Returns: inverse-filtered B-polarization healpy alm array
        """
        tfname = os.path.join(self.lib_dir, 'sim_%04d_blm.fits'%idx  if idx >= 0 else 'dat_blm.fits')
        if not os.path.exists(tfname):
            if self.soltn_lib is None:
                soltn = None
            else:
                soltn = np.array([self.soltn_lib.get_sim_emliklm(idx), self.soltn_lib.get_sim_bmliklm(idx)])
            elm, blm = self._apply_ivf_p(self.sim_lib.get_pmap(idx), soltn=soltn)
            if self.cache:
                hp.write_alm(tfname, blm, overwrite=True)
                hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_elm.fits'%idx if idx >= 0 else 'dat_elm.fits'), elm, overwrite=True)
        else:
            blm = hp.read_alm(tfname)
        if self.lfilt is not None:
            hp.almxfl(blm, self.lfilt, inplace=True)
        return blm


    def get_sim_tmliklm(self, idx):
        """Returns a Wiener-filtered temperature simulation.
            Args: idx: simulation index
            Returns: Wiener-filtered temperature healpy alm array
        """
        cltt = self.cl['tt'][:len(self.lfilt)]*self.lfilt if self.lfilt is not None else self.cl['tt']
        return hp.almxfl(self.get_sim_tlm(idx), cltt)

    def get_sim_emliklm(self, idx):
        """Returns a Wiener-filtered E-polarization simulation.
            Args: idx: simulation index
            Returns: Wiener-filtered E-polarization healpy alm array
        """
        clee = self.cl['ee'][:len(self.lfilt)]*self.lfilt if self.lfilt is not None else self.cl['ee']
        return hp.almxfl(self.get_sim_elm(idx), clee)

    def get_sim_bmliklm(self, idx):
        """Returns a Wiener-filtered B-polarization simulation.
            Args: idx: simulation index
            Returns: Wiener-filtered B-polarization healpy alm array
        """
        clbb = self.cl['bb'][:len(self.lfilt)]*self.lfilt if self.lfilt is not None else self.cl['bb']
        return hp.almxfl(self.get_sim_blm(idx), clbb)

    def get_sim_tlmivf(self, idx):
        """Returns an inverse variance temperature simulation.
            Args: idx: simulation index
            Returns: Wiener-filtered temperature healpy alm array
        """
        print("Returning inverse variance filtered tlm")
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl['tt'])
        return hp.almxfl(self.get_sim_tlm(idx), fl)

    def get_sim_eblmivf(self, idx):
        """Returns an inverse variance filtered E-polarization simulation.
            Args: idx: simulation index
            Returns: Wiener-filtered E-polarization healpy alm array
        """
        print("Returning inverse variance filtered elm")
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl['ee'])
        elm,blm = self.get_sim_eblm(idx)
        return hp.almxfl(elm, fl), hp.almxfl(blm, fl)

    def get_sim_elmivf(self, idx):
        """Returns an inverse variance filtered E-polarization simulation.
            Args: idx: simulation index
            Returns: Wiener-filtered E-polarization healpy alm array
        """
        print("Returning inverse variance filtered elm")
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl['ee'])
        return hp.almxfl(self.get_sim_elm(idx), fl)

    def get_sim_blmivf(self, idx):
        """Returns an inverse variance B-polarization simulation.
            Args: idx: simulation index
            Returns: Wiener-filtered B-polarization healpy alm array
        """
        print("Returning inverse variance filtered blm")
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl['bb'])
        return hp.almxfl(self.get_sim_blm(idx), fl)


class library_cinv_sepTP(library_sepTP):
    """Library to perform inverse-variance filtering of a simulation library.

        Suitable for separate temperature and polarization filtering.

        Args:
            lib_dir (str): a
            sim_lib: simulation library instance (requires get_sim_tmap, get_sim_pmap methods)
            cinvt: temperature-only filtering library
            cinvp: poalrization-only filtering library
            soltn_lib (optional): simulation libary providing starting guesses for the filtering.

    """

    def __init__(self, lib_dir, sim_lib, cinvt:cinv_t, cinvp:cinv_p, 
                 cl_weights:dict, soltn_lib=None, lfilt=None):
        self.cinv_t = cinvt
        self.cinv_p = cinvp
        super(library_cinv_sepTP, self).__init__(lib_dir, sim_lib, cl_weights, 
                                                 soltn_lib=soltn_lib,
                                                 lfilt=lfilt)
        Y=0
        if Y==0:
        #if mpi.rank == 0:
            fname_mask = os.path.join(self.lib_dir, "fmask.fits.gz")
            if not os.path.exists(fname_mask):
                fmask = self.cinv_t.get_fmask()

                #assert np.all(fmask == self.cinv_p.get_fmask())
                hp.write_map(fname_mask, fmask, overwrite=True)

        #mpi.barrier()
        #cinv_utils.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict())

    def hashdict(self):
        return {'cinv_t': self.cinv_t.hashdict(),
                'cinv_p': self.cinv_p.hashdict()
                #,
                #'sim_lib': self.sim_lib.hashdict()
                }

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))

    def get_tal(self, a, lmax=None):
        assert (a.lower() in ['t', 'e', 'b']), a
        if a.lower() == 't':
            return self.cinv_t.get_tal(a, lmax=lmax)
        else:
            return self.cinv_p.get_tal(a, lmax=lmax)

    def get_ftl(self, lmax=None):
        """Isotropic approximation to temperature inverse variance filtering.
            :math:`F^{T}_\ell = (C_\ell^{TT} + N^{T}_\ell / b_\ell^2)^{-1}`
        """
        return self.cinv_t.get_ftl(lmax=lmax)

    def get_fel(self, lmax=None):
        """Isotropic approximation to E-polarization inverse variance filtering.
            :math:`F^{E}_\ell = (C_\ell^{EE} + N^{E}_\ell / b_\ell^2)^{-1}`
        """
        return self.cinv_p.get_fel(lmax=lmax)

    def get_fbl(self, lmax=None):
        """Isotropic approximation to B-polarization inverse variance filtering.
            :math:`F^{B}_\ell = (C_\ell^{BB} + N^{B}_\ell / b_\ell^2)^{-1}`
        """
        return self.cinv_p.get_fbl(lmax=lmax)

    def _apply_ivf_t(self, tmap, soltn=None):
        return self.cinv_t.apply_ivf(tmap, soltn=soltn)

    def _apply_ivf_p(self, pmap, soltn=None):
        return self.cinv_p.apply_ivf(pmap, soltn=soltn)

    '''
    def get_tmliklm(self, idx):
        return  hp.almxfl(self.get_sim_tlm(idx), self.cinv_t.cl['tt'])

    def get_emliklm(self, idx):
        assert not hasattr(self.cinv_p.cl, 'eb')
        return  hp.almxfl(self.get_sim_elm(idx), self.cinv_t.cl['ee'])

    def get_bmliklm(self, idx):
        assert not hasattr(self.cinv_p.cl, 'eb')
        return  hp.almxfl(self.get_sim_blm(idx), self.cinv_t.cl['bb'])
    '''

