"""
Classes holding C-inv related object for healpix maps
Copied from what is implemented by Kimmy in spt3g_software
https://github.com/SouthPoleTelescope/spt3g_software/blob/curvlens/lensing/python/cinv_hp.py
but with additional cleaning/formatting and commenting.
"""

import os, sys
import healpy as hp
import numpy as np
import pickle as pk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
import utils, hp_utils, cinv_utils
import opfilt_hp_p, opfilt_hp_t, opfilt_hp_tp, cd_solve, cd_monitors


class cinv(object):
    def __init__(self, lib_dir, lmax, eps_min, use_mpi=False):
        self.lib_dir = lib_dir  # Output directory
        self.lmax = lmax  # Lmax to use for filtering
        self.eps_min = eps_min  # Tolerance

        if use_mpi == True:
            from cinv_utils import mpi

    def get_tal(self, a, lmax=None):
        if lmax is None:
            lmax = self.lmax
        assert a.lower() in ["t", "e", "b"], a
        ret = np.loadtxt(os.path.join(self.lib_dir, "tal.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[: lmax + 1]

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))

    def get_ftl(self, lmax=None):
        if lmax is None:
            lmax = self.lmax
        ret = np.loadtxt(os.path.join(self.lib_dir, "ftl.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[: lmax + 1]

    def get_fel(self, lmax=None):
        if lmax is None:
            lmax = self.lmax
        ret = np.loadtxt(os.path.join(self.lib_dir, "fel.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[: lmax + 1]

    def get_fbl(self, lmax=None):
        if lmax is None:
            lmax = self.lmax
        ret = np.loadtxt(os.path.join(self.lib_dir, "fbl.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[: lmax + 1]

    def solve(self, soltn, tpn_map):
        finifunc = getattr(self.opfilt, "calc_fini")
        self.iter_tot = 0
        self.prev_eps = None
        dot_op = self.opfilt.DotOperator()
        logger = cd_monitors.logger_basic

        tpn_alm = self.opfilt.calc_prep(tpn_map, self.s_inv_filt, self.n_inv_filt)
        monitor = cd_monitors.MonitorBasic(
            dot_op, logger=logger, iter_max=np.inf, eps_min=self.eps_min
        )

        fwd_op = self.opfilt.ForwardOperator(self.s_inv_filt, self.n_inv_filt)

        pre_op = self.opfilt.PreOperatorDiag(
            self.s_inv_filt.s_cls, self.n_inv_filt, nl_res=self.s_inv_filt.nl_res
        )

        cd_solve.cd_solve(
            soltn,
            b=tpn_alm,
            fwd_op=fwd_op,
            pre_ops=[pre_op],
            dot_op=dot_op,
            criterion=monitor,
            tr=cinv_utils.cd_solve.tr_cg,
            cache=cinv_utils.cd_solve.CacheMemory(),
        )

        finifunc(soltn, self.s_inv_filt, self.n_inv_filt)


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

    def __init__(
        self,
        lib_dir,
        lmax,
        nside,
        cl,
        nl_res,
        ninv,
        tf1d,
        tf2d,
        eps_min=1.0e-5,
        ellscale=True,
    ):
        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (
            lib_dir,
            lmax,
            nside,
        )
        assert isinstance(ninv, list)
        super(cinv_t, self).__init__(lib_dir, lmax, eps_min)

        print("Initializing cinv_t")

        # Scaling factor
        ell = np.arange(lmax + 1, dtype=float)

        if ellscale:
            print('Applying ellscaling l(l+1)/2pi')
            rescal_cl = np.sqrt(ell * (ell + 1) / 2.0 / np.pi)
            rescal_cl[0] = 1
        else:
            print('Not applying any scaling')
            rescal_cl = np.ones_like(ell)

        self.rescal_cl = rescal_cl

        # rescaled cls (Dls by default)
        dl = {k: rescal_cl**2 * cl[k][: lmax + 1] for k in cl.keys()}

        # Do not scale nl_res here. Forward model has 1/(cltt+nltt/tf1d^2).
        # cltt has l^2 and tf^2 has 1/l^2 factor already.
        # This means nltt/tf1d_scal^2 == l^2 * nltt/tf1d_unscal^2.
        # nl_res    = {k: rescal_cl ** 2 * nl_res[k][:lmax + 1] for k in  nl_res.keys()}

        tf1d = tf1d[: lmax + 1] * cinv_utils.cli(rescal_cl)
        tf2d = hp.almxfl(tf2d, cinv_utils.cli(rescal_cl))

        self.nside = nside

        self.cl = cl
        self.dl = dl
        self.nl_res = nl_res
        self.ninv = ninv

        self.tf1d = tf1d
        self.tf2d = tf2d

        # Set up s_inv_filt and n_inv_filt
        self.s_inv_filt = hp_utils.jit(
            opfilt_hp_t.SkyInverseFilter, dl, nl_res, lmax, tf1d, tf2d=tf2d
        )
        self.n_inv_filt = hp_utils.jit(
            opfilt_hp_t.NoiseInverseFilter, ninv, tf1d, tf2d=tf2d
        )
        self.opfilt = opfilt_hp_t

    def _calc_ftl(self):
        ninv = self.n_inv_filt.n_inv
        npix = len(ninv[:])
        NlevT_uKamin = (
            np.sqrt(4.0 * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0]))
            * 180.0
            * 60.0
            / np.pi
        )
        print("cinv_t::noiseT_uk_arcmin = %.3f" % NlevT_uKamin)

        s_cls = self.cl
        tf1d = self.tf1d

        ftl = cinv_utils.cli(
            s_cls["tt"][0 : self.lmax + 1]
            + (NlevT_uKamin * np.pi / 180.0 / 60.0) ** 2 / tf1d[0 : self.lmax + 1] ** 2
        )
        ftl[0:2] = 0.0

        return ftl

    def apply_ivf(self, tmap, soltn=None):
        if soltn is None:
            talm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex128)
        else:
            talm = soltn.copy()
        self.solve(talm, tmap)
        hp.almxfl(talm, self.rescal_cl, inplace=True)
        return talm


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

    def __init__(
        self,
        lib_dir,
        lmax,
        nside,
        cl,
        nl_res,
        ninv,
        tf1dE,
        tf1dB,
        tf2dE,
        tf2dB,
        eps_min=1.0e-5,
        ellscale=True,
    ):
        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (
            lib_dir,
            lmax,
            nside,
        )
        assert isinstance(ninv, list)
        super(cinv_p, self).__init__(lib_dir, lmax, eps_min)

        print("Initializing cinv_p")

        # Scaling factor
        ell = np.arange(lmax + 1, dtype=float)

        if ellscale:
            print('Applying ellscaling l(l+1)/2pi')
            rescal_cl = np.sqrt(ell * (ell + 1) / 2.0 / np.pi)
            rescal_cl[0] = 1
        else:
            print('Not applying any scaling')
            rescal_cl = np.ones_like(ell)

        self.rescal_cl = rescal_cl

        # rescaled cls (Dls by default)
        dl = {k: rescal_cl**2 * cl[k][: lmax + 1] for k in cl.keys()}

        # Do not scale nl_res here. Forward model has 1/(cltt+nltt/tf1d^2).
        # cltt has l^2 and tf^2 has 1/l^2 factor already.
        # This means nltt/tf1d_scal^2 == l^2 * nltt/tf1d_unscal^2.
        # nl_res    = {k: rescal_cl ** 2 * nl_res[k][:lmax + 1] for k in  nl_res.keys()}

        tf1dE = tf1dE[: lmax + 1] * cinv_utils.cli(rescal_cl)
        tf1dB = tf1dB[: lmax + 1] * cinv_utils.cli(rescal_cl)
        
        tf2dE = hp.almxfl(tf2dE, cinv_utils.cli(rescal_cl))
        tf2dB = hp.almxfl(tf2dB, cinv_utils.cli(rescal_cl))

        self.nside = nside
        self.cl = cl
        self.dl = dl
        self.tf1dE = tf1dE
        self.tf1dB = tf1dB
        
        self.tf2dE = tf2dE
        self.tf2dB = tf2dB
        self.ninv = ninv
        self.nl_res = nl_res

        # Set up s_inv_filt and n_inv_filt
        self.s_inv_filt = hp_utils.jit(
            opfilt_hp_p.SkyInverseFilter, dl, nl_res, lmax, tf1dE, tf1dB, tf2dE, tf2dB  
        )
        self.n_inv_filt = hp_utils.jit(
            opfilt_hp_p.NoiseInverseFilter, ninv, tf1dE, tf1dB, tf2dE, tf2dB
        )
        self.opfilt = opfilt_hp_p

    def apply_ivf(self, tmap, soltn=None):
        if soltn is not None:
            print("soltn is not None")
            assert len(soltn) == 2
            assert hp.Alm.getlmax(soltn[0].size) == self.lmax, (
                hp.Alm.getlmax(soltn[0].size),
                self.lmax,
            )
            assert hp.Alm.getlmax(soltn[1].size) == self.lmax, (
                hp.Alm.getlmax(soltn[1].size),
                self.lmax,
            )
            talm = hp_utils.eblm([soltn[0], soltn[1]])
        else:
            print("soltn is None")
            telm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex128)
            tblm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex128)
            talm = hp_utils.eblm([telm, tblm])

        assert len(tmap) == 2
        print("cinv_p.solve")
        self.solve(talm, [tmap[0], tmap[1]])
        hp.almxfl(talm.elm, self.rescal_cl, inplace=True)
        hp.almxfl(talm.blm, self.rescal_cl, inplace=True)
        return talm.elm, talm.blm

    def _calc_febl(self):
        assert not "eb" in self.cl.keys()

        if len(self.ninv) == 1:
            ninv = self.n_inv_filt.n_inv[0]
            npix = len(ninv)
            NlevP_uKamin = (
                np.sqrt(
                    4.0 * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0])
                )
                * 180.0
                * 60.0
                / np.pi
            )
        else:
            assert len(self.ninv) == 3
            ninv = self.n_inv_filt.n_inv
            NlevP_uKamin = (
                0.5
                * np.sqrt(
                    4.0
                    * np.pi
                    / len(ninv[0])
                    / np.sum(ninv[0])
                    * len(np.where(ninv[0] != 0.0)[0])
                )
                * 180.0
                * 60.0
                / np.pi
            )
            NlevP_uKamin += (
                0.5
                * np.sqrt(
                    4.0
                    * np.pi
                    / len(ninv[2])
                    / np.sum(ninv[2])
                    * len(np.where(ninv[2] != 0.0)[0])
                )
                * 180.0
                * 60.0
                / np.pi
            )

        print("cinv_p::noiseP_uk_arcmin = %.3f" % NlevP_uKamin)

        s_cls = self.cl
        tf1dE = self.n_inv_filt.tf1dE
        tf1dB = self.n_inv_filt.tf1dB
        
        fel = cinv_utils.cli(
            s_cls["ee"][: self.lmax + 1]
            + (NlevP_uKamin * np.pi / 180.0 / 60.0) ** 2
            * cinv_utils.cli(tf1dE[0 : self.lmax + 1] ** 2)
        )
        fbl = cinv_utils.cli(
            s_cls["bb"][: self.lmax + 1]
            + (NlevP_uKamin * np.pi / 180.0 / 60.0) ** 2
            * cinv_utils.cli(tf1dB[0 : self.lmax + 1] ** 2)
        )

        fel[0:2] *= 0.0
        fbl[0:2] *= 0.0

        return fel, fbl

    def _calc_tal(self):
        return cinv_utils.cli(self.tf1dE)

    def _calc_mask(self):
        mask = np.ones(hp.nside2npix(self.nside), dtype=float)
        for ninv in self.ninv:
            assert hp.npix2nside(len(ninv)) == self.nside
            mask *= ninv > 0.0
        return mask


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

    def __init__(
        self,
        lib_dir,
        lmax,
        nside,
        cl,
        nl_res,
        ninv,
        tf1d_t,
        tf1d_p,
        tf2d_t,
        tf2d_p,
        eps_min=1.0e-5,
        ellscale=False,
    ):
        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (
            lib_dir,
            lmax,
            nside,
        )
        assert len(ninv) == 2 or len(ninv) == 4  # TT, (QQ + UU)/2 or TT,QQ,QU,UU
        super(cinv_tp, self).__init__(lib_dir, lmax, eps_min)

        print("Initializing cinv_tp")

        # Scaling factor
        ell = np.arange(lmax + 1, dtype=float)

        if ellscale:
            print('Applying ell scaling: l(l+1)/2pi')
            rescal_cl = np.sqrt(ell * (ell + 1) / 2.0 / np.pi)
            rescal_cl[0] = 1
        else:
            print('Not applying any ell scaling')
            rescal_cl = np.ones_like(ell)

        self.rescal_cl = rescal_cl

        # rescaled cls (Dls by default)
        dl = {k: rescal_cl**2 * cl[k][: lmax + 1] for k in cl.keys()}

        # Do not scale nl_res here. Forward model has 1/(cltt+nltt/tf1d^2).
        # cltt has l^2 and tf^2 has 1/l^2 factor already.
        # This means nltt/tf1d_scal^2 == l^2 * nltt/tf1d_unscal^2.
        # nl_res    = {k: rescal_cl ** 2 * nl_res[k][:lmax + 1] for k in  nl_res.keys()}

        tf1d_t = tf1d_t[: lmax + 1] * cinv_utils.cli(rescal_cl)
        tf1d_p = tf1d_p[: lmax + 1] * cinv_utils.cli(rescal_cl)

        tf2d_t = hp.almxfl(tf2d_t, cinv_utils.cli(rescal_cl))
        tf2d_p = hp.almxfl(tf2d_p, cinv_utils.cli(rescal_cl))

        self.nside = nside
        self.cl = cl
        self.dl = dl
        self.tf1d_t = tf1d_t
        self.tf1d_p = tf1d_p
        self.tf2d_t = tf2d_t
        self.tf2d_p = tf2d_p
        self.ninv = ninv
        self.nl_res = nl_res

        self.s_inv_filt = hp_utils.jit(
            opfilt_hp_tp.SkyInverseFilter,
            dl,
            nl_res,
            lmax,
            tf1d_t,
            tf1d_p,
            tf2d_t,
            tf2d_p,
        )

        self.n_inv_filt = hp_utils.jit(
            opfilt_hp_tp.NoiseInverseFilter, ninv, tf1d_t, tf1d_p, tf2d_t, tf2d_p
        )

        self.opfilt = opfilt_hp_tp

    def apply_ivf(self, tqumap, soltn=None):  # , apply_fini=''):
        assert len(tqumap) == 3
        if soltn is not None:
            ttlm, telm, tblm = soltn
        else:
            ttlm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex128)
            telm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex128)
            tblm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex128)

        talm = hp_utils.teblm([ttlm, telm, tblm])
        self.solve(talm, [tqumap[0], tqumap[1], tqumap[2]])  # , apply_fini=apply_fini)
        hp.almxfl(talm.tlm, self.rescal_cl, inplace=True)
        hp.almxfl(talm.elm, self.rescal_cl, inplace=True)
        hp.almxfl(talm.blm, self.rescal_cl, inplace=True)
        return talm.tlm, talm.elm, talm.blm


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

    def __init__(
        self, lib_dir, sim_lib, cl_weights, lfilt=None, soltn_lib=None, add_noise=False
    ):
        self.lib_dir = lib_dir
        self.sim_lib = sim_lib
        self.cl = cl_weights
        self.lfilt = lfilt
        self.add_noise = add_noise

        self.soltn_lib = soltn_lib

    def get_sim_teblm(self, idx):
        """
        Returns an inverse-filtered T/E/B simulation.

        Args: idx    : simulation index
              Returns: inverse-filtered temperature healpy alm array
        """
        tfname = os.path.join(
            self.lib_dir, "sim_%04d_tlm.fits" % idx if idx >= 0 else "dat_tlm.fits"
        )

        if not os.path.exists(tfname):
            pass
        else:
            print("Loading file: %s" % tfname)
            tlm, elm, blm = hp.read_alm(tfname, hdu=[1, 2, 3])

        if self.lfilt is not None:
            hp.almxfl(tlm, self.lfilt, inplace=True)
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)

        return tlm, elm, blm

    def get_sim_tlm(self, idx):
        """
        Returns an inverse-filtered temperature simulation.

        Args: idx    : simulation index
              Returns: inverse-filtered temperature healpy alm array
        """
        tfname = os.path.join(
            self.lib_dir, "sim_%04d_tlm.fits" % idx if idx >= 0 else "dat_tlm.fits"
        )
        if not os.path.exists(tfname):
            print("tlm file doesnt exit so creating one")
            tlm = self._apply_ivf_t(
                self.sim_lib.get_tmap(idx, add_noise=self.add_noise),
                soltn=None
                if self.soltn_lib is None
                else self.soltn_lib.get_sim_tmliklm(idx),
            )
        else:
            print("Loading file: %s" % tfname)
            tlm = hp.read_alm(tfname)
        if self.lfilt is not None:
            hp.almxfl(tlm, self.lfilt, inplace=True)
        return tlm

    def get_sim_elm(self, idx):
        """Returns an inverse-filtered E-polarization simulation.
        Args:idx: simulation index
        Returns: inverse-filtered E-polarization healpy alm array
        """
        print("library_sepTP.get_sim_elm")
        tfname = os.path.join(
            self.lib_dir, "sim_%04d_elm.fits" % idx if idx >= 0 else "dat_elm.fits"
        )
        Y = 0
        if Y == 0:
            print("Creating new")
            if self.soltn_lib is None:
                soltn = None
            else:
                soltn = np.array(
                    [
                        self.soltn_lib.get_sim_emliklm(idx),
                        self.soltn_lib.get_sim_bmliklm(idx),
                    ]
                )

            elm, blm = self._apply_ivf_p(
                self.sim_lib.get_pmap(idx, add_noise=self.add_noise), soltn=soltn
            )

        else:
            sys.exit("Failed to load alm")

        if self.lfilt is not None:
            hp.almxfl(elm, self.lfilt, inplace=True)
        return elm

    def get_sim_blm(self, idx):
        """Returns an inverse-filtered B-polarization simulation.
        Args: idx: simulation index
        Returns: inverse-filtered B-polarization healpy alm array
        """
        tfname = os.path.join(
            self.lib_dir, "sim_%04d_blm.fits" % idx if idx >= 0 else "dat_blm.fits"
        )
        if not os.path.exists(tfname):
            if self.soltn_lib is None:
                soltn = None
            else:
                soltn = np.array(
                    [
                        self.soltn_lib.get_sim_emliklm(idx),
                        self.soltn_lib.get_sim_bmliklm(idx),
                    ]
                )
            elm, blm = self._apply_ivf_p(self.sim_lib.get_pmap(idx), soltn=soltn)
            if self.cache:
                hp.write_alm(tfname, blm, overwrite=True)
                hp.write_alm(
                    os.path.join(
                        self.lib_dir,
                        "sim_%04d_elm.fits" % idx if idx >= 0 else "dat_elm.fits",
                    ),
                    elm,
                    overwrite=True,
                )
        else:
            blm = hp.read_alm(tfname)
        if self.lfilt is not None:
            hp.almxfl(blm, self.lfilt, inplace=True)
        return blm

    def get_sim_teblm(self, idx):
        """
        Returns an inverse-filtered T/E/B simulation.

        Args: idx    : simulation index
              Returns: inverse-filtered temperature healpy alm array
        """
        tfname = os.path.join(
            self.lib_dir, "sim_%04d_tlm.fits" % idx if idx >= 0 else "dat_tlm.fits"
        )

        # Loading unfiltered alms
        if not os.path.exists(tfname):
            pass
        else:
            print("Loading file: %s" % tfname)
            tlm, elm, blm = hp.read_alm(tfname, hdu=[1, 2, 3])

        # Apply lmin/lmax cuts in 1d
        if self.lfilt is not None:
            hp.almxfl(tlm, self.lfilt, inplace=True)
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)

        return tlm, elm, blm

    def get_sim_eblm(self, idx):
        """Returns an inverse-filtered E-polarization simulation.
        Args:idx: simulation index
        Returns: inverse-filtered E-polarization healpy alm array
        """
        if self.soltn_lib is None:
            soltn = None
        else:
            soltn = np.array(
                [
                    self.soltn_lib.get_sim_emliklm(idx),
                    self.soltn_lib.get_sim_bmliklm(idx),
                ]
            )

        elm, blm = self._apply_ivf_p(
            self.sim_lib.get_pmap(idx, add_noise=self.add_noise), soltn=soltn
        )

        if self.lfilt is not None:
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)
        return elm, blm

    def get_sim_tmliklm(self, idx):
        """Returns a Wiener-filtered temperature simulation.
        Args: idx: simulation index
        Returns: Wiener-filtered temperature healpy alm array
        """
        cltt = (
            self.cl["tt"][: len(self.lfilt)] * self.lfilt
            if self.lfilt is not None
            else self.cl["tt"]
        )
        return hp.almxfl(self.get_sim_tlm(idx), cltt)

    def get_sim_emliklm(self, idx):
        """Returns a Wiener-filtered E-polarization simulation.
        Args: idx: simulation index
        Returns: Wiener-filtered E-polarization healpy alm array
        """
        print("library_sepTP.get_sim_emliklm")
        clee = (
            self.cl["ee"][: len(self.lfilt)] * self.lfilt
            if self.lfilt is not None
            else self.cl["ee"]
        )
        return hp.almxfl(self.get_sim_elm(idx), clee)

    def get_sim_bmliklm(self, idx):
        """Returns a Wiener-filtered B-polarization simulation.
        Args: idx: simulation index
        Returns: Wiener-filtered B-polarization healpy alm array
        """
        clbb = (
            self.cl["bb"][: len(self.lfilt)] * self.lfilt
            if self.lfilt is not None
            else self.cl["bb"]
        )
        return hp.almxfl(self.get_sim_blm(idx), clbb)

    def get_sim_tlmivf(self, idx):
        """Returns an inverse variance temperature simulation.
        Args: idx: simulation index
        Returns: Wiener-filtered temperature healpy alm array
        """
        print("Returning inverse variance filtered tlm")
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["tt"])
        return hp.almxfl(self.get_sim_tlm(idx), fl)

    def get_sim_eblmivf(self, idx):
        """Returns an inverse variance filtered E-polarization simulation.
        Args: idx: simulation index
        Returns: Wiener-filtered E-polarization healpy alm array
        """
        print(
            "library_sepTP.get_sim_eblmivf -- Returning inverse variance filtered eblm"
        )
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["ee"])
        elm, blm = self.get_sim_eblm(idx)
        return hp.almxfl(elm, fl), hp.almxfl(blm, fl)


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

    def __init__(
        self,
        lib_dir,
        sim_lib,
        cinvt,
        cinvp,
        cl_weights,
        soltn_lib=None,
        lfilt=None,
    ):
        self.cinv_t = cinvt
        self.cinv_p = cinvp
        super(library_cinv_sepTP, self).__init__(
            lib_dir, sim_lib, cl_weights, soltn_lib=soltn_lib, lfilt=lfilt
        )

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))

    def get_tal(self, a, lmax=None):
        assert a.lower() in ["t", "e", "b"], a
        if a.lower() == "t":
            return self.cinv_t.get_tal(a, lmax=lmax)
        else:
            return self.cinv_p.get_tal(a, lmax=lmax)

    def get_ftl(self, lmax=None):
        return self.cinv_t.get_ftl(lmax=lmax)

    def get_fel(self, lmax=None):
        return self.cinv_p.get_fel(lmax=lmax)

    def get_fbl(self, lmax=None):
        return self.cinv_p.get_fbl(lmax=lmax)

    def _apply_ivf_t(self, tmap, soltn=None):
        return self.cinv_t.apply_ivf(tmap, soltn=soltn)

    def _apply_ivf_p(self, pmap, soltn=None):
        return self.cinv_p.apply_ivf(pmap, soltn=soltn)



class library_jTP(object):
    """Template class for CMB inverse-variance and Wiener-filtering library.

    This one is suitable whenever the temperature and polarization maps are jointly filtered.

    Args:
        lib_dir (str): directory where hashes and filtered maps will be cached.
        sim_lib : simulation library instance. *sim_lib* must have *get_sim_tmap* and *get_sim_pmap* methods.
        cl_weights: CMB spectra (gCMB or lCMB), used to compute the Wiener-filtered CMB from the inverse variance filtered maps.

    """

    def __init__(self, lib_dir, sim_lib, cl_weights, lfilt=None, soltn_lib=None):
        # assert np.all([k in cl_weights.keys() for k in ['tt', 'ee', 'bb']])
        self.lib_dir = lib_dir
        self.sim_lib = sim_lib
        self.cl = cl_weights
        self.lfilt = lfilt
        self.soltn_lib = soltn_lib


    def _get_alms(self, a, idx):
        assert a in ["t", "e", "b", "teb"]
        tfname = os.path.join(
            self.lib_dir, "sim_%04d_tlm.fits" % idx if idx >= 0 else "dat_tlm.fits"
        )
        fname = tfname.replace("tlm.fits", a + "lm.fits")
        if not os.path.exists(fname):
            T = self.sim_lib.get_tmap(idx)
            Q, U = self.sim_lib.get_pmap(idx)
            if self.soltn_lib is None:
                soltn = None
            else:
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

    def get_sim_teblm(self, idx):
        return self._get_alms("teb", idx)

    def get_sim_eblm(self, idx):
        elm = self._get_alms("e", idx)
        blm = self._get_alms("b", idx)
        return elm, blm

    def get_sim_tlm(self, idx):
        return self._get_alms("t", idx)

    def get_sim_elm(self, idx):
        return self._get_alms("e", idx)

    def get_sim_blm(self, idx):
        return self._get_alms("b", idx)

    def get_sim_tmliklm(self, idx):
        ret = hp.almxfl(self.get_sim_tlm(idx), self.cl["tt"])
        for k in ["te", "tb"]:
            cl = self.cl.get(k[0] + k[1], self.cl.get(k[1] + k[0], None))
            if cl is not None:
                ret += hp.almxfl(self._get_alms(k[1], idx), cl)
        return ret

    def get_sim_emliklm(self, idx):
        ret = hp.almxfl(self.get_sim_elm(idx), self.cl["ee"])
        for k in ["et", "eb"]:
            cl = self.cl.get(k[0] + k[1], self.cl.get(k[1] + k[0], None))
            if cl is not None:
                ret += hp.almxfl(self._get_alms(k[1], idx), cl)
        return ret

    def get_sim_bmliklm(self, idx):
        ret = hp.almxfl(self.get_sim_blm(idx), self.cl["bb"])
        for k in ["bt", "be"]:
            cl = self.cl.get(k[0] + k[1], self.cl.get(k[1] + k[0], None))
            if cl is not None:
                ret += hp.almxfl(self._get_alms(k[1], idx), cl)
        return ret

    def get_sim_tlmivf(self, idx):
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["tt"])
        return hp.almxfl(self.get_sim_tlm(idx), fl)

    def get_sim_elmivf(self, idx):
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["ee"])
        return hp.almxfl(self.get_sim_elm(idx), fl)

    def get_sim_blmivf(self, idx):
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["bb"])
        return hp.almxfl(self.get_sim_blm(idx), fl)

    def get_sim_eblmivf(self, idx):
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["ee"])
        elm, blm = self.get_sim_eblm(idx)
        return hp.almxfl(elm, fl), hp.almxfl(blm, fl)

    def get_sim_teblmivf(self, idx):
        fl = self.lfilt if self.lfilt is not None else np.ones_like(self.cl["ee"])
        tlm, elm, blm = self.get_sim_teblm(idx)
        return hp.almxfl(tlm, fl), hp.almxfl(elm, fl), hp.almxfl(blm, fl)


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

    def __init__(
        self,
        lib_dir,
        sim_lib,
        cinv_jtp: cinv_tp,
        cl_weights: dict,
        soltn_lib=None,
        lfilt=None,
    ):
        self.cinv_tp = cinv_jtp
        super(library_cinv_jTP, self).__init__(
            lib_dir, sim_lib, cl_weights, soltn_lib=soltn_lib, lfilt=lfilt
        )

    def get_fal(self, lmax=None):
        return self.cinv_tp.get_fal(lmax=lmax)

    def _apply_ivf(self, tqumap, soltn=None):
        return self.cinv_tp.apply_ivf(tqumap, soltn=soltn)
