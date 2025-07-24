"""
Classes holding C-inv related object for healpix maps
Copied from what is implemented by Kimmy in spt3g_software
https://github.com/SouthPoleTelescope/spt3g_software/blob/curvlens/lensing/python/cinv_hp.py
but with additional cleaning/formatting and commenting.
"""

import os
import sys
import healpy as hp
import numpy as np
import logging
from . import opfilt_hp
from . import cd_solve, cd_monitors
from . import hp_utils, cinv_utils
logger = logging.getLogger(__name__)


class cinv(object):
    def __init__(self, lib_dir, lmax, nside, cl, nl_res, eps_min, ellscale, tf, g=None):
        """
        Parameters
        ----------
        lib_dir: str
            Directory where intermediate data will be cached.
        lmax: int
        nside: int
        cl: dict
            Dictionary of CMB power spectra, including 'tt', 'ee', 'bb', 'te'.
        nl_res:
        eps_min: float
        ellscale: bool
            If True, scale Cl as Dl.
        tf: opfilt_hp.TFObj
            Transfer function object
        g: Geometry=None
            The `ducc` wrapper object used to speed up spherical harmonic transforms. If specified, the SHT will
            only be applied on relevant rings, i.e., an implicit binary mask is applied.  If None, a full-sky
            Geometry object of `nside` will be used. This should give identical results as healpy but still a 2x
            speed-up.
        """
        assert lmax >= 1024
        assert nside >= 512
        self.lib_dir = lib_dir  # Output directory
        self.lmax = lmax  # Lmax to use for filtering
        self.nside = nside
        self.eps_min = eps_min  # Tolerance

        logger.info(f"Initializing {self.__class__.__name__}")

        # Scaling factor
        ell = np.arange(lmax + 1, dtype=float)
        if ellscale:
            logger.debug("Applying ell scaling: l(l+1)/2pi")
            rescal_cl = np.sqrt(ell * (ell + 1) / 2.0 / np.pi)
            rescal_cl[0] = 1
        else:
            logger.debug("Not applying any ell scaling")
            rescal_cl = np.ones_like(ell)
        self.rescal_cl = rescal_cl

        self.tf = tf
        invfac = cinv_utils.cli(self.rescal_cl)
        self.tf *= invfac

        self.cl = cl.copy()
        # rescaled cls (Dls by default)
        # Do not scale nl_res here. Forward model has 1/(cltt+nltt/tf1d^2).
        # cltt has l^2 and tf^2 has 1/l^2 factor already.
        # This means nltt/tf1d_scal^2 == l^2 * nltt/tf1d_unscal^2.
        # nl_res    = {k: rescal_cl ** 2 * nl_res[k][:lmax + 1] for k in  nl_res.keys()}
        self.dl = {k: rescal_cl**2 * v[:lmax+1] for k, v in cl.items()}

        self.nl_res = nl_res.copy()
        if self.nl_res is None:
            self.nl_res = {k: np.zeros_like(v[:lmax+1]) for k, v in cl.items()}
        else:
            for k, v in self.nl_res.items():
                self.nl_res[k][:2] = 0  # enforce this cutoff is important for convergence!

        self.s_inv_filt = None
        self.n_inv_filt = None

        self.iter_tot = None
        self.prev_eps = None
        self._pre_op = None  # placeholder for pre_op cache

        self.eps = None  # tracks the convergence history

        if g is None:
            from healqest.ducc_sht import Geometry
            self.g = Geometry(nside=nside, dec_range=None)  # fallback to full-sky ducc
            # self.g = None  # fallback to hp
        else:
            self.g = g
            assert self.g.nside == nside

    def get_tal(self, a, lmax=None):
        if lmax is None:
            lmax = self.lmax
        assert a.lower() in ["t", "e", "b"], a
        ret = np.loadtxt(os.path.join(self.lib_dir, "tal.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[: lmax + 1]

    def solve(self, soltn, tpn_map, verbose=True):
        self.iter_tot = 0
        self.prev_eps = None
        dot_op = opfilt_hp.DotOperator()
        cd_logger = cd_monitors.logger_basic() if verbose else cd_monitors.logger_none()

        tpn_alm = self.n_inv_filt.calc_prep(tpn_map)
        monitor = cd_monitors.MonitorBasic(dot_op, cd_logger=cd_logger, iter_max=np.inf, eps_min=self.eps_min)
        fwd_op = opfilt_hp.ForwardOperator(self.s_inv_filt, self.n_inv_filt)

        cd_solve.cd_solve(
            soltn,
            b=tpn_alm,
            fwd_op=fwd_op,
            pre_ops=[self.pre_op],
            dot_op=dot_op,
            criterion=monitor,
            tr=cd_solve.tr_cg,
            cache=cd_solve.CacheMemory(),
        )
        self.eps = monitor.eps
        opfilt_hp.calc_fini(soltn, self.s_inv_filt)

    def get_fl(self, pol, lmax):
        pass

    @property
    def pre_op(self):
        if self._pre_op is None:
            self._pre_op = opfilt_hp.PreOperatorDiag(self.s_inv_filt.s_cls, self.n_inv_filt, tf=self.tf,
                                                     nl_res=self.s_inv_filt.nl_res)
        return self._pre_op


class cinv_t(cinv):
    """
    Polarization-only inverse-variance (or Wiener-) filtering instance.

    This class performs polarization-only filtering of CMB maps using inverse-variance
    (or Wiener-) filtering. The implementation supports template projection.

    Parameters
    ----------
    lib_dir : str
        Directory where masks and other ancillary data will be cached.
    lmax : int
        Maximum multipole at which the filtered alm's are reconstructed.
    nside : int
        Healpy resolution of the maps to be filtered.
    cl : dict
        Fiducial CMB power spectra used for filtering.
    nl_res : array_like
        Dictionary of noise residual used in the filtering.
        (Additional details on the expected format should be provided.)
    ninv: list of np.array or str
        Inverse pixel variance maps.
    tf1d : array_like
        1d transfer function.
    tf2d : array_like
        2d trasnfer function.
    eps_min : float, optional
        Minimum epsilon value for the filtering procedure, by default 1.0e-5.
    ellscale : bool, optional
        Whether to scale by multipole ell, by default True.

    """

    def __init__(self, lib_dir, lmax, nside, cl, nl_res, ninv, tf1d, tf2d=None, bl=None, lx_cut=None,
                 eps_min=1.0e-5, ellscale=True, g=None, mmin=None):
        assert len(ninv) == 1
        tf = opfilt_hp.TFObj(npol=1, lmax=lmax, tf1d=tf1d, tf2d=tf2d, lx_cut=lx_cut, bl=bl, m_cut=mmin)
        # only take the first entry as the temperation ninv
        super(cinv_t, self).__init__(lib_dir, lmax, nside=nside, cl=cl, nl_res=nl_res, eps_min=eps_min,
                                     ellscale=ellscale, tf=tf, g=g)

        # Set up s_inv_filt and n_inv_filt
        self.s_inv_filt = hp_utils.jit(opfilt_hp.SkyInverseFilter, s_cls=self.dl, nl_res=self.nl_res, tf=self.tf)
        self.n_inv_filt = hp_utils.jit(opfilt_hp.NoiseInverseFilter, n_inv=ninv, tf=self.tf, g=self.g)

    def apply_ivf(self, tmap, soltn=None):
        if soltn is None:
            talm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex128)
        else:
            talm = soltn.copy()
        logger.info("cinv_t.solve")
        self.solve(talm, tmap)
        hp.almxfl(talm, self.rescal_cl, inplace=True)
        return talm

    def get_fl(self, pol, lmax):
        assert pol.lower()=='t'
        out = self.pre_op.fl[0, :self.lmax + 1] * self.rescal_cl[:self.lmax + 1] ** 2
        if lmax is None:
            return out
        else:
            return out[:lmax+1]


class cinv_p(cinv):
    """
    Polarization-only inverse-variance (or Wiener-) filtering instance.

    This class performs polarization-only filtering of CMB maps using inverse-variance
    (or Wiener-) filtering. The implementation supports template projection.

    Parameters
    ----------
    lib_dir : str
        Directory where masks and other ancillary data will be cached.
    lmax : int
        Maximum multipole at which the filtered alm's are reconstructed.
    nside : int
        Healpy resolution of the maps to be filtered.
    cl : dict
        Fiducial CMB power spectra used for filtering.
    nl_res : array_like
        Dictionary of noise residual used in the filtering.
        (Additional details on the expected format should be provided.)
    ninv : list
        Inverse pixel variance maps. Must be a list containing either 3 elements
        (for QQ, QU, and UU noise) or 1 element (for QQ = UU noise). Each element
        can be a file path or a Healpy map, and they must be consistent with the given nside.
    tf1d : array_like
        1d transfer function.
    tf2d : array_like
        2d trasnfer function.
    eps_min : float, optional
        Minimum epsilon value for the filtering procedure, by default 1.0e-5.
    ellscale : bool, optional
        Whether to scale by multipole ell, by default True.

    """

    def __init__(self, lib_dir, lmax, nside, cl, nl_res, ninv, tf1d, tf2d=None, bl=None, lx_cut=None,
                 eps_min=1.0e-5, ellscale=True, g=None, mmin=None):

        assert isinstance(ninv, list)
        assert len(ninv) in [2]
        tf = opfilt_hp.TFObj(npol=2, lmax=lmax, tf1d=tf1d, tf2d=tf2d, lx_cut=lx_cut, bl=bl, m_cut=mmin)
        super(cinv_p, self).__init__(lib_dir, lmax, nside=nside, cl=cl, nl_res=nl_res, eps_min=eps_min,
                                     ellscale=ellscale, tf=tf, g=g)

        # Set up s_inv_filt and n_inv_filt
        self.s_inv_filt = hp_utils.jit(opfilt_hp.SkyInverseFilter, s_cls=self.dl, nl_res=self.nl_res, tf=self.tf)

        self.n_inv_filt = hp_utils.jit(opfilt_hp.NoiseInverseFilter, n_inv=ninv, tf=self.tf, g=self.g)

    def apply_ivf(self, qumap, soltn=None):
        if soltn is not None:
            logger.debug("soltn is not None")
            assert len(soltn) == 2
            assert hp.Alm.getlmax(soltn.shape[-1]) == self.lmax
            eblm = soltn.copy()
        else:
            logger.debug("soltn is None")
            eblm = np.zeros((2, hp.Alm.getsize(self.lmax)), dtype=np.complex128)

        assert len(qumap) == 2
        logger.info("cinv_p.solve")
        self.solve(eblm, qumap)
        hp.almxfl(eblm[0], self.rescal_cl, inplace=True)
        hp.almxfl(eblm[1], self.rescal_cl, inplace=True)
        return eblm

    # def _calc_febl(self):
    #     assert "eb" not in self.cl.keys()
    #
    #     if len(self.ninv) == 1:
    #         ninv = self.n_inv_filt.n_inv[0]
    #         npix = len(ninv)
    #         NlevP_uKamin = (
    #             np.sqrt(
    #                 4.0 * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0])
    #             )
    #             * 180.0
    #             * 60.0
    #             / np.pi
    #         )
    #     else:
    #         assert len(self.ninv) == 3
    #         ninv = self.n_inv_filt.n_inv
    #         NlevP_uKamin = (
    #             0.5
    #             * np.sqrt(
    #                 4.0
    #                 * np.pi
    #                 / len(ninv[0])
    #                 / np.sum(ninv[0])
    #                 * len(np.where(ninv[0] != 0.0)[0])
    #             )
    #             * 180.0
    #             * 60.0
    #             / np.pi
    #         )
    #         NlevP_uKamin += (
    #             0.5
    #             * np.sqrt(
    #                 4.0
    #                 * np.pi
    #                 / len(ninv[2])
    #                 / np.sum(ninv[2])
    #                 * len(np.where(ninv[2] != 0.0)[0])
    #             )
    #             * 180.0
    #             * 60.0
    #             / np.pi
    #         )
    #
    #     logger.debug(f"cinv_p::noiseP_uk_arcmin = {NlevP_uKamin:.3f}")
    #
    #     s_cls = self.cl
    #     tf1dE = self.n_inv_filt.tf1dE
    #     tf1dB = self.n_inv_filt.tf1dB
    #
    #     fel = cinv_utils.cli(
    #         s_cls["ee"][: self.lmax + 1]
    #         + (NlevP_uKamin * np.pi / 180.0 / 60.0) ** 2
    #         * cinv_utils.cli(tf1dE[0: self.lmax + 1] ** 2)
    #     )
    #     fbl = cinv_utils.cli(
    #         s_cls["bb"][: self.lmax + 1]
    #         + (NlevP_uKamin * np.pi / 180.0 / 60.0) ** 2
    #         * cinv_utils.cli(tf1dB[0: self.lmax + 1] ** 2)
    #     )
    #
    #     fel[0:2] *= 0.0
    #     fbl[0:2] *= 0.0
    #
    #     return fel, fbl
    #
    # def _calc_tal(self):
    #     return cinv_utils.cli(self.tf1dE)
    #
    # def _calc_mask(self):
    #     mask = np.ones(hp.nside2npix(self.nside), dtype=float)
    #     for ninv in self.ninv:
    #         assert hp.npix2nside(len(ninv)) == self.nside
    #         mask *= ninv > 0.0
    #     return mask

    def get_fl(self, pol, lmax):
        assert pol.lower() in 'eb'
        i = 'eb'.index(pol.lower())
        out = self.pre_op.fl[i, :self.lmax+1]*self.rescal_cl[:self.lmax+1]**2
        if lmax is None:
            return out
        else:
            return out[:lmax+1]


class cinv_tp(cinv):
    """
    Initialize a polarization-only inverse-variance (or Wiener-) filtering instance with
    separate transfer functions for temperature and polarization.

    This constructor sets up the filtering instance for processing CMB maps using
    inverse-variance filtering. In this version, separate
    transfer functions are provided for temperature and polarization channels in both
    one-dimensional (1d) and two-dimensional (2d) forms.

    Parameters
    ----------
    lib_dir : str
        Directory where masks and other ancillary data will be cached.
    lmax : int
        Maximum multipole at which the filtered alm's are reconstructed.
    nside : int
        Healpy resolution of the maps to be filtered.
    cl : dict
        Fiducial CMB power spectra used for filtering.
    nl_res : array_like
        Dictionary of noise residuals used in the filtering.
        (Additional details on the expected format should be provided.)
    ninv : list
        Inverse pixel variance maps. Must be a list containing either 3 elements
        (for QQ, QU, and UU noise) or 1 element (for QQ = UU noise). Each element
        can be a file path or a Healpy map, and they must be consistent with the given nside.
    tf1d_t : array_like
        One-dimensional transfer function for temperature.
    tf1d_p : array_like
        One-dimensional transfer function for polarization.
    tf2d_t : array_like
        Two-dimensional transfer function for temperature.
    tf2d_p : array_like
        Two-dimensional transfer function for polarization.
    eps_min : float, optional
        Minimum epsilon value for the filtering procedure, by default 1.0e-5.
    ellscale : bool, optional
        Whether to scale by multipole ell, by default True.
    """

    def __init__(self, lib_dir, lmax, nside, cl, nl_res, ninv, tf1d, tf2d, bl=None, lx_cut=None,
                 eps_min=1.0e-5, ellscale=False, g=None, mmin=None):

        assert isinstance(ninv, list)
        assert len(ninv) in [3]  # TT/PP or TT/QQ/UU
        tf = opfilt_hp.TFObj(npol=3, lmax=lmax, tf1d=tf1d, tf2d=tf2d, lx_cut=lx_cut, bl=bl, m_cut=mmin)

        super(cinv_tp, self).__init__(lib_dir, lmax, nside=nside, cl=cl, nl_res=nl_res, eps_min=eps_min,
                                      ellscale=ellscale, g=g, tf=tf)
        self.n_inv_filt = hp_utils.jit(opfilt_hp.NoiseInverseFilter, ninv, tf=self.tf, g=g)
        self.s_inv_filt = hp_utils.jit(opfilt_hp.SkyInverseFilterJoint, s_cls=self.dl, nl_res=nl_res, tf=self.tf)

    @property
    def pre_op(self):
        if self._pre_op is None:
            self._pre_op = opfilt_hp.PreOperatorDiagJoint(self.s_inv_filt.s_cls, self.n_inv_filt, tf=self.tf,
                                                          nl_res=self.s_inv_filt.nl_res)
        return self._pre_op

    def apply_ivf(self, tqumap, soltn=None):  # , apply_fini=''):
        assert len(tqumap) == 3
        if soltn is not None:
            ttlm, telm, tblm = soltn
        else:
            teblm = np.zeros((3, hp.Alm.getsize(self.lmax)), dtype=np.complex128)
        self.solve(teblm, [tqumap[0], tqumap[1], tqumap[2]])  # , apply_fini=apply_fini)
        hp.almxfl(teblm[0], self.rescal_cl, inplace=True)
        hp.almxfl(teblm[1], self.rescal_cl, inplace=True)
        hp.almxfl(teblm[2], self.rescal_cl, inplace=True)
        return teblm

    def get_fl(self, pol, lmax):
        assert pol.lower() in 'teb'
        i = 'teb'.index(pol.lower())
        out = self.pre_op.fl[i, :self.lmax+1]*self.rescal_cl[:self.lmax+1]**2
        if lmax is None:
            return out
        else:
            return out[:lmax+1]


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

    def __init__(self, lib_dir, sim_lib, cl_weights, lfilt=None, soltn_lib=None, add_noise=False):
        self.lib_dir = lib_dir
        self.sim_lib = sim_lib
        self.cl = cl_weights
        self.lfilt = lfilt
        self.add_noise = add_noise
        self.soltn_lib = soltn_lib
        self.g = None

    def get_sim_teblm(self, idx):
        """
        Returns an inverse-filtered T/E/B simulation.

        Args: idx    : simulation index
              Returns: inverse-filtered temperature healpy alm array
        """
        tfname = os.path.join(self.lib_dir, "sim_%04d_tlm.fits" % idx if idx >= 0 else "dat_tlm.fits")

        if not os.path.exists(tfname):
            pass
        else:
            logger.info("Loading file: %s" % tfname)
            tlm, elm, blm = hp.read_alm(tfname, hdu=[1, 2, 3])

        if self.lfilt is not None:
            hp.almxfl(tlm, self.lfilt, inplace=True)
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)

        return tlm, elm, blm

    def get_sim_tlm(self, seed, cmbid, bundle):
        """
        Returns an inverse-filtered temperature simulation.

        Args: idx    : simulation index
              Returns: inverse-filtered temperature healpy alm array
        """
        tfname = os.path.join(self.lib_dir, f"sim_{seed:04d}_{cmbid}_tlm.fits" if seed >= 0 else "dat_tlm.fits")
        if not os.path.exists(tfname):
            logger.info("tlm file doesnt exit so creating one")
            if self.soltn_lib is not None:
                soltn = self.soltn_lib.get_sim_tmliklm(seed, cmbid)
            else:
                soltn = None
            map_in = self.sim_lib.get_tmap(seed, cmbid, bundle=bundle, add_noise=self.add_noise, g=self.g)
            tlm = self._apply_ivf_t(map_in, soltn=soltn)
        else:
            logger.info(f"Loading file: {tfname}")
            tlm = hp.read_alm(tfname)
        if self.lfilt is not None:
            hp.almxfl(tlm, self.lfilt, inplace=True)
        return tlm

    # def get_sim_elm(self, idx):
    #     """Returns an inverse-filtered E-polarization simulation.
    #     Args:idx: simulation index
    #     Returns: inverse-filtered E-polarization healpy alm array
    #     """
    #     logger.info("library_sepTP.get_sim_elm")
    #     tfname = os.path.join(self.lib_dir, "sim_%04d_elm.fits" % idx if idx >= 0 else "dat_elm.fits")
    #     Y = 0
    #     if Y == 0:
    #         logger.info("Creating new")
    #         if self.soltn_lib is None:
    #             soltn = None
    #         else:
    #             soltn = np.array([self.soltn_lib.get_sim_emliklm(idx),
    #                               self.soltn_lib.get_sim_bmliklm(idx)])
    #
    #         elm, blm = self._apply_ivf_p(self.sim_lib.get_pmap(idx, add_noise=self.add_noise), soltn=soltn)
    #     else:
    #         sys.exit("Failed to load alm")
    #
    #     if self.lfilt is not None:
    #         hp.almxfl(elm, self.lfilt, inplace=True)
    #     return elm
    #
    # def get_sim_blm(self, idx):
    #     """Returns an inverse-filtered B-polarization simulation.
    #     Args: idx: simulation index
    #     Returns: inverse-filtered B-polarization healpy alm array
    #     """
    #     tfname = os.path.join(
    #         self.lib_dir, "sim_%04d_blm.fits" % idx if idx >= 0 else "dat_blm.fits"
    #     )
    #     if not os.path.exists(tfname):
    #         if self.soltn_lib is None:
    #             soltn = None
    #         else:
    #             soltn = np.array(
    #                 [
    #                     self.soltn_lib.get_sim_emliklm(idx),
    #                     self.soltn_lib.get_sim_bmliklm(idx),
    #                 ]
    #             )
    #         elm, blm = self._apply_ivf_p(self.sim_lib.get_pmap(idx), soltn=soltn)
    #         if self.cache:
    #             hp.write_alm(tfname, blm, overwrite=True)
    #             hp.write_alm(
    #                 os.path.join(
    #                     self.lib_dir,
    #                     "sim_%04d_elm.fits" % idx if idx >= 0 else "dat_elm.fits",
    #                 ),
    #                 elm,
    #                 overwrite=True,
    #             )
    #     else:
    #         blm = hp.read_alm(tfname)
    #     if self.lfilt is not None:
    #         hp.almxfl(blm, self.lfilt, inplace=True)
    #     return blm
    #
    # def get_sim_teblm(self, idx):
    #     """
    #     Returns an inverse-filtered T/E/B simulation.
    #
    #     Args: idx    : simulation index
    #           Returns: inverse-filtered temperature healpy alm array
    #     """
    #     tfname = os.path.join(self.lib_dir, "sim_%04d_tlm.fits" % idx if idx >= 0 else "dat_tlm.fits")
    #
    #     Loading unfiltered alms
        # if not os.path.exists(tfname):
        #     pass
        # else:
        #     logger.info("Loading file: %s" % tfname)
        #     tlm, elm, blm = hp.read_alm(tfname, hdu=[1, 2, 3])
        #
        # Apply lmin/lmax cuts in 1d
        # if self.lfilt is not None:
        #     hp.almxfl(tlm, self.lfilt, inplace=True)
        #     hp.almxfl(elm, self.lfilt, inplace=True)
        #     hp.almxfl(blm, self.lfilt, inplace=True)
        #
        # return tlm, elm, blm

    def get_sim_eblm(self, seed, cmbid, bundle):
        """Returns an inverse-filtered E-polarization simulation.

        Parameters
        ----------
        seed: int
        cmbid: int

        Returns
        -------
        elm, blm
            inverse-filtered E/B alm arrays
        """
        if self.soltn_lib is None:
            soltn = None
        else:
            raise NotImplementedError
            # soltn = np.array([self.soltn_lib.get_sim_emliklm(idx),
            #                   self.soltn_lib.get_sim_bmliklm(idx),])

        map_in = self.sim_lib.get_pmap(seed, cmbid, bundle=bundle, add_noise=self.add_noise, g=self.g)
        elm, blm = self._apply_ivf_p(map_in, soltn=soltn)

        if self.lfilt is not None:
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)
        return elm, blm


    # def get_sim_tmliklm(self, idx):
    #     '''
    #     Returns a Wiener-filtered temperature simulation.
    #     Args: idx: simulation index
    #     Returns: Wiener-filtered temperature healpy alm array
    #     '''
    #     cltt = (
    #         self.cl["tt"][: len(self.lfilt)] * self.lfilt
    #         if self.lfilt is not None
    #         else self.cl["tt"]
    #     )
    #     return hp.almxfl(self.get_sim_tlm(idx), cltt)
    #
    # def get_sim_emliklm(self, idx):
    #     '''Returns a Wiener-filtered E-polarization simulation.
    #     Args: idx: simulation index
    #     Returns: Wiener-filtered E-polarization healpy alm array
    #     ''''''
    #     logger.info("library_sepTP.get_sim_emliklm")
    #     clee = (
    #         self.cl["ee"][: len(self.lfilt)] * self.lfilt
    #         if self.lfilt is not None
    #         else self.cl["ee"]
    #     )
    #     return hp.almxfl(self.get_sim_elm(idx), clee)
    #
    # def get_sim_bmliklm(self, idx):
    #     """Returns a Wiener-filtered B-polarization simulation.
    #     Args: idx: simulation index
    #     Returns: Wiener-filtered B-polarization healpy alm array
    #     """
    #     clbb = (
    #         self.cl["bb"][: len(self.lfilt)] * self.lfilt
    #         if self.lfilt is not None
    #         else self.cl["bb"]
    #     )
    #     return hp.almxfl(self.get_sim_blm(idx), clbb)
    #
    # def get_sim_tlmivf(self, idx):
    #     """Returns an inverse variance temperature simulation.
    #     Args: idx: simulation index
    #     Returns: Wiener-filtered temperature healpy alm array
    #     """
    #     logger.info("Returning inverse variance filtered tlm")
    #     fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["tt"])
    #     return hp.almxfl(self.get_sim_tlm(idx), fl)
    #
    # def get_sim_eblmivf(self, idx):
    #     """Returns an inverse variance filtered E-polarization simulation.
    #     Args: idx: simulation index
    #     Returns: Wiener-filtered E-polarization healpy alm array
    #     """
    #     logger.info("library_sepTP.get_sim_eblmivf -- Returning inverse variance filtered eblm")
    #     fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["ee"])
    #     elm, blm = self.get_sim_eblm(idx)
    #     return hp.almxfl(elm, fl), hp.almxfl(blm, fl)


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

    def __init__(self, lib_dir, sim_lib, cinvt=None, cinvp=None, cl_weights=None, soltn_lib=None, lfilt=None,
                 add_noise=False):
        super(library_cinv_sepTP, self).__init__(lib_dir, sim_lib, cl_weights, soltn_lib=soltn_lib, lfilt=lfilt,
                                                 add_noise=add_noise)
        self.cinv_t = cinvt
        self.cinv_p = cinvp
        self.g = cinvt.g

    # def get_fmask(self):
    #     return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))
    #
    # def get_tal(self, a, lmax=None):
    #     assert a.lower() in ["t", "e", "b"], a
    #     if a.lower() == "t":
    #         return self.cinv_t.get_tal(a, lmax=lmax)
    #     else:
    #         return self.cinv_p.get_tal(a, lmax=lmax)

    def get_fl(self, pol, lmax=None):
        if pol == 't':
            return self.cinv_t.get_fl(pol='t', lmax=lmax)
        elif pol in 'eb':
            return self.cinv_p.get_fl(pol=pol, lmax=lmax)
        else:
            raise ValueError("pol must be 't'/'e'/'b'")

    def _apply_ivf_t(self, tmap, soltn=None):
        return self.cinv_t.apply_ivf(tmap, soltn=soltn)

    def _apply_ivf_p(self, pmap, soltn=None):
        return self.cinv_p.apply_ivf(pmap, soltn=soltn)

    def get_eps(self, ):
        out = []
        if self.cinv_t is not None:
            out += list(self.cinv_t.eps)
        if self.cinv_p is not None:
            out += list(self.cinv_p.eps)
        return np.array(out)


class library_jTP(object):
    """Template class for CMB inverse-variance and Wiener-filtering library.

    This one is suitable whenever the temperature and polarization maps are jointly filtered.

    Args:
        lib_dir (str): directory where hashes and filtered maps will be cached.
        sim_lib : simulation library instance. *sim_lib* must have *get_sim_tmap* and *get_sim_pmap* methods.
        cl_weights: CMB spectra (gCMB or lCMB), used to compute the Wiener-filtered CMB from the inverse variance filtered maps.

    """

    def __init__(self, lib_dir, sim_lib, cl_weights, lfilt=None, soltn_lib=None, add_noise=False):
        # assert np.all([k in cl_weights.keys() for k in ['tt', 'ee', 'bb']])
        self.lib_dir = lib_dir
        self.sim_lib = sim_lib
        self.cl = cl_weights
        self.lfilt = lfilt
        self.soltn_lib = soltn_lib
        self.add_noise = add_noise
        self.g = None

    def _get_alms(self, a, seed, cmbid, bundle):
        assert a in ["t", "e", "b", "teb"]

        if True:
            T = self.sim_lib.get_tmap(seed, cmbid, bundle=bundle, add_noise=self.add_noise, g=self.g)
            Q, U = self.sim_lib.get_pmap(seed, cmbid, bundle=bundle, add_noise=self.add_noise, g=self.g)
            if self.soltn_lib is None:
                soltn = None
            else:
                raise NotImplementedError
                tlm = self.soltn_lib.get_sim_tmliklm(idx)
                elm = self.soltn_lib.get_sim_emliklm(idx)
                blm = self.soltn_lib.get_sim_bmliklm(idx)
                soltn = (tlm, elm, blm)
            tlm, elm, blm = self._apply_ivf([T, Q, U], soltn=soltn)

        else:
            tlm, elm, blm = hp.read_alm(tfname, hdu=[1, 2, 3])

        if self.lfilt is not None:
            hp.almxfl(tlm, self.lfilt, inplace=True)
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)

        return {"teb": (tlm, elm, blm), "t": tlm, "e": elm, "b": blm}[a]

    def get_sim_teblm(self, seed, cmbid, bundle):
        return self._get_alms("teb", seed, cmbid, bundle=bundle)

    # def get_sim_eblm(self, idx):
    #     elm = self._get_alms("e", idx)
    #     blm = self._get_alms("b", idx)
    #     return elm, blm
    #
    # def get_sim_tlm(self, idx):
    #     return self._get_alms("t", idx)
    #
    # def get_sim_elm(self, idx):
    #     return self._get_alms("e", idx)
    #
    # def get_sim_blm(self, idx):
    #     return self._get_alms("b", idx)
    #
    # def get_sim_tmliklm(self, idx):
    #     ret = hp.almxfl(self.get_sim_tlm(idx), self.cl["tt"])
    #     for k in ["te", "tb"]:
    #         cl = self.cl.get(k[0] + k[1], self.cl.get(k[1] + k[0], None))
    #         if cl is not None:
    #             ret += hp.almxfl(self._get_alms(k[1], idx), cl)
    #     return ret
    #
    # def get_sim_emliklm(self, idx):
    #     ret = hp.almxfl(self.get_sim_elm(idx), self.cl["ee"])
    #     for k in ["et", "eb"]:
    #         cl = self.cl.get(k[0] + k[1], self.cl.get(k[1] + k[0], None))
    #         if cl is not None:
    #             ret += hp.almxfl(self._get_alms(k[1], idx), cl)
    #     return ret
    #
    # def get_sim_bmliklm(self, idx):
    #     ret = hp.almxfl(self.get_sim_blm(idx), self.cl["bb"])
    #     for k in ["bt", "be"]:
    #         cl = self.cl.get(k[0] + k[1], self.cl.get(k[1] + k[0], None))
    #         if cl is not None:
    #             ret += hp.almxfl(self._get_alms(k[1], idx), cl)
    #     return ret
    #
    # def get_sim_tlmivf(self, idx):
    #     fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["tt"])
    #     return hp.almxfl(self.get_sim_tlm(idx), fl)
    #
    # def get_sim_elmivf(self, idx):
    #     fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["ee"])
    #     return hp.almxfl(self.get_sim_elm(idx), fl)
    #
    # def get_sim_blmivf(self, idx):
    #     fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["bb"])
    #     return hp.almxfl(self.get_sim_blm(idx), fl)
    #
    # def get_sim_eblmivf(self, idx):
    #     fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["ee"])
    #     elm, blm = self.get_sim_eblm(idx)
    #     return hp.almxfl(elm, fl), hp.almxfl(blm, fl)
    #
    # def get_sim_teblmivf(self, idx):
    #     fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["ee"])
    #     tlm, elm, blm = self.get_sim_teblm(idx)
    #     return hp.almxfl(tlm, fl), hp.almxfl(elm, fl), hp.almxfl(blm, fl)


class library_cinv_jTP(library_jTP):
    """Library to perform inverse-variance filtering of a simulation library.

    Suitable for joint temperature and polarization filtering.

    Args:
        lib_dir (str): a place to cache the maps
        sim_lib: simulation library instance (requires get_sim_tmap, get_sim_pmap methods)
        cinv_jtp: temperature and pol joint filtering library
        cl_weights: spectra used to build the Wiener filtered leg from the inverse-variance maps
        soltn_lib (optional): simulation libary providing starting guesses for the filtering.


    """

    def __init__(self, lib_dir, sim_lib, cinv_jtp: cinv_tp, cl_weights: dict=None, soltn_lib=None, lfilt=None,
                 add_noise=False):

        super(library_cinv_jTP, self).__init__(lib_dir, sim_lib, cl_weights, soltn_lib=soltn_lib, lfilt=lfilt,
                                               add_noise=add_noise)
        self.cinv_tp = cinv_jtp
        self.g = cinv_jtp.g

    # def get_fal(self, lmax=None):
        # return self.cinv_tp.get_fal(lmax=lmax)

    def _apply_ivf(self, tqumap, soltn=None):
        return self.cinv_tp.apply_ivf(tqumap, soltn=soltn)

    def get_fl(self, pol, lmax=None):
        return self.cinv_tp.get_fl(pol=pol, lmax=lmax)

    def get_eps(self, ):
        out = []
        if self.cinv_tp is not None:
            out += list(self.cinv_tp.eps)
        return np.array(out)
