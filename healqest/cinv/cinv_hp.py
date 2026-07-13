"""
Classes holding C-inv related object for healpix maps.

Copied from what is implemented by Kimmy in spt3g_software
https://github.com/SouthPoleTelescope/spt3g_software/blob/curvlens/lensing/python/cinv_hp.py
but with additional cleaning/formatting and commenting.
"""

import abc
import healpy as hp
import numpy as np
from .. import log
from . import opfilt_hp
from . import cd_solve, cd_monitors
from . import hp_utils, cinv_utils
from .opfilt_hp import SkyInverseFilterJoint, SkyInverseFilter, TFObj

logger = log.get_logger(__name__)


class cinv(abc.ABC):
    lmax: int

    def __init__(self, lmax, nside, cl, nl_res, eps_min, ellscale, tf, g=None):
        """
        Base class for inverse-variance filtering.

        Parameters
        ----------
        lmax: int
        nside: int
        cl: dict
            Dictionary of CMB power spectra, including 'tt', 'ee', 'bb', 'te'.
        nl_res:
        eps_min: float
        ellscale: bool
            If True, scale Cl as Dl.
        tf: TFObj
            Transfer function object
        g: Geometry=None
        """
        assert lmax >= 1024
        assert nside >= 512
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
        self.rescal_cl = np.array(rescal_cl)

        self.tf = tf
        invfac = cinv_utils.cli(self.rescal_cl)
        self.tf *= invfac

        self.cl = cl.copy()
        # rescaled cls (Dls by default)
        # Do not scale nl_res here. Forward model has 1/(cltt+nltt/tf1d^2).
        # cltt has l^2 and tf^2 has 1/l^2 factor already.
        # This means nltt/tf1d_scal^2 == l^2 * nltt/tf1d_unscal^2.
        # nl_res    = {k: rescal_cl ** 2 * nl_res[k][:lmax + 1] for k in  nl_res.keys()}
        self.dl = {k: rescal_cl**2 * v[: lmax + 1] for k, v in cl.items()}

        self.nl_res = nl_res.copy()
        if self.nl_res is None:
            self.nl_res = {k: np.zeros_like(v[: lmax + 1]) for k, v in cl.items()}
        else:
            for k, v in self.nl_res.items():
                self.nl_res[k][:2] = 0  # enforce this cutoff is important for convergence!

        self.s_inv_filt = None
        self.n_inv_filt = None

        self.prev_eps = None
        self._pre_op = None  # placeholder for pre_op cache

        self.eps = None  # tracks the convergence history

        if g is None:
            from healqest.ducc_sht import Geometry

            self.g = Geometry(nside=nside)  # fallback to full-sky ducc
        else:
            self.g = g
            assert self.g.nside == nside

    def solve(self, soltn, tpn_map, verbose=True):
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

    def set_ninv_model(self, ninv, ninv_nl=None):
        from .opfilt_hp import NoiseInverseFilterAlm, NoiseInverseFilterPix

        if ninv_nl is None:
            self.n_inv_filt = hp_utils.jit(NoiseInverseFilterPix, n_inv=ninv, tf=self.tf, g=self.g)
        else:
            self.n_inv_filt = hp_utils.jit(
                NoiseInverseFilterAlm, n_inv=ninv, tf=self.tf, g=self.g, ninv_nl=ninv_nl
            )

    @abc.abstractmethod
    def get_fl(self, pol, lmax):
        raise NotImplementedError("get_fl method must be implemented in subclass")

    @property
    def pre_op(self):
        if self._pre_op is None:
            self._pre_op = opfilt_hp.PreOperatorDiag(
                self.s_inv_filt.s_cls, self.n_inv_filt, tf=self.tf, nl_res=self.s_inv_filt.nl_res
            )
        return self._pre_op


class cinv_t(cinv):
    """
    Polarization-only inverse-variance (or Wiener-) filtering instance.

    This class performs polarization-only filtering of CMB maps using inverse-variance
    (or Wiener-) filtering. The implementation supports template projection.

    Parameters
    ----------
    lmax : int
        Maximum multipole at which the filtered alm's are reconstructed.
    nside : int
        Healpy resolution of the maps to be filtered.
    cl : dict
        Fiducial CMB power spectra used for filtering.
    nl_res : np.array
        Dictionary of noise residual used in the filtering.
        (Additional details on the expected format should be provided.)
    ninv: list of np.array or str.
        Inverse pixel variance maps.
    tf1d : np.ndarray or list
        1d transfer function.
    tf2d : np.ndarray or list, optional.
        2d trasnfer function.
    eps_min : float=1e-5
        Minimum epsilon value for the filtering procedure, by default 1.0e-5.
    ellscale : bool=True
        Whether to scale by multipole ell, by default True.

    """

    def __init__(
        self,
        lmax,
        nside,
        cl,
        nl_res,
        ninv,
        tf1d,
        tf2d=None,
        bl=None,
        mtheta=None,
        eps_min=1.0e-5,
        ellscale=True,
        g=None,
        mmin=None,
        ninv_nl=None,
    ):
        assert len(ninv) == 1
        tf = TFObj(1, lmax=lmax, tf1d=tf1d, bl=bl, tf2d=tf2d, mtheta=mtheta, m_cut=mmin)
        # only take the first entry as the temperation ninv
        super().__init__(
            lmax, nside=nside, cl=cl, nl_res=nl_res, eps_min=eps_min, ellscale=ellscale, tf=tf, g=g
        )

        # Set up s_inv_filt and n_inv_filt
        self.s_inv_filt = hp_utils.jit(SkyInverseFilter, s_cls=self.dl, nl_res=self.nl_res, tf=self.tf)
        self.set_ninv_model(ninv=ninv, ninv_nl=ninv_nl)

    def apply_ivf(self, tmap, soltn=None):
        if soltn is None:
            talm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex128)
        else:
            talm = soltn.copy()
        logger.info("cinv_t.solve")
        self.solve(talm, tmap)
        # noinspection PyTypeChecker
        hp.almxfl(talm, self.rescal_cl, inplace=True)
        return talm

    def get_fl(self, pol, lmax):
        assert pol.lower() == 't'
        out = self.pre_op.fl[0, : self.lmax + 1] * self.rescal_cl[: self.lmax + 1] ** 2
        if lmax is None:
            return out
        else:
            return out[: lmax + 1]


class cinv_p(cinv):
    """
    Polarization-only inverse-variance (or Wiener-) filtering instance.

    This class performs polarization-only filtering of CMB maps using inverse-variance
    (or Wiener-) filtering. The implementation supports template projection.

    Parameters
    ----------
    lmax : int
        Maximum multipole at which the filtered alm's are reconstructed.
    nside : int
        Healpy resolution of the maps to be filtered.
    cl : dict
        Fiducial CMB power spectra used for filtering.
    nl_res : np.ndarray
        Dictionary of noise residual used in the filtering.
        (Additional details on the expected format should be provided.)
    ninv : list
        Inverse pixel variance maps. Must be a list containing either 3 elements
        (for QQ, QU, and UU noise) or 1 element (for QQ = UU noise). Each element
        can be a file path or a Healpy map, and they must be consistent with the given nside.
    tf1d : np.ndarray or list
        1d transfer function.
    tf2d : np.ndarray or list, optional
        2d trasnfer function.
    eps_min : float=1e-5
        Minimum epsilon value for the filtering procedure, by default 1.0e-5.
    ellscale : bool=True
        Whether to scale by multipole ell, by default True.

    """

    def __init__(
        self,
        lmax,
        nside,
        cl,
        nl_res,
        ninv,
        tf1d,
        tf2d=None,
        bl=None,
        mtheta=None,
        eps_min=1.0e-5,
        ellscale=True,
        g=None,
        mmin=None,
        ninv_nl=None,
    ):
        assert isinstance(ninv, list)
        assert len(ninv) in [2]
        tf = TFObj(2, lmax=lmax, tf1d=tf1d, bl=bl, tf2d=tf2d, mtheta=mtheta, m_cut=mmin)
        super().__init__(
            lmax, nside=nside, cl=cl, nl_res=nl_res, eps_min=eps_min, ellscale=ellscale, tf=tf, g=g
        )

        # Set up s_inv_filt and n_inv_filt
        self.s_inv_filt = hp_utils.jit(SkyInverseFilter, s_cls=self.dl, nl_res=self.nl_res, tf=self.tf)
        self.set_ninv_model(ninv=ninv, ninv_nl=ninv_nl)

    def apply_ivf(self, qumap, soltn=None):
        if soltn is not None:
            raise NotImplementedError
        else:
            eblm = np.zeros((2, hp.Alm.getsize(self.lmax)), dtype=np.complex128)

        assert len(qumap) == 2
        logger.info("cinv_p.solve")
        self.solve(eblm, qumap)
        # noinspection PyTypeChecker
        hp.almxfl(eblm[0], self.rescal_cl, inplace=True)
        # noinspection PyTypeChecker
        hp.almxfl(eblm[1], self.rescal_cl, inplace=True)
        return eblm

    def get_fl(self, pol, lmax):
        assert pol.lower() in 'eb'
        i = 'eb'.index(pol.lower())
        out = self.pre_op.fl[i, : self.lmax + 1] * self.rescal_cl[: self.lmax + 1] ** 2
        if lmax is None:
            return out
        else:
            return out[: lmax + 1]


class cinv_tp(cinv):
    """
    Joint temperature and polarization cinv-filtering, for GMV.

    Parameters
    ----------
    lmax : int
        Maximum multipole at which the filtered alm's are reconstructed.
    nside : int
        Healpy resolution of the maps to be filtered.
    cl : dict
        Fiducial CMB power spectra used for filtering.
    nl_res : np.ndarray
        Dictionary of noise residuals used in the filtering.
        (Additional details on the expected format should be provided.)
    ninv : list
        Inverse pixel variance maps. Must be a list containing either 3 elements
        (for QQ, QU, and UU noise) or 1 element (for QQ = UU noise). Each element
        can be a file path or a Healpy map, and they must be consistent with the given nside.
    tf1d : np.ndarray or list
        One-dimensional transfer function for temperature/polarization.
    tf2d : np.ndarray or list, optional.
        Two-dimensional transfer function for temperatur/polarization.
    eps_min : float=1e-5
        Minimum epsilon value for the filtering procedure, by default 1.0e-5.
    ellscale : bool=False
        Whether to scale by multipole ell, by default True.
    """

    def __init__(
        self,
        lmax,
        nside,
        cl,
        nl_res,
        ninv,
        tf1d,
        tf2d=None,
        bl=None,
        mtheta=None,
        eps_min=1.0e-5,
        ellscale=False,
        g=None,
        mmin=None,
        ninv_nl=None,
    ):
        assert isinstance(ninv, list)
        assert len(ninv) in [2]  # TT/PP
        tf = TFObj(3, lmax=lmax, tf1d=tf1d, bl=bl, tf2d=tf2d, mtheta=mtheta, m_cut=mmin)

        super().__init__(
            lmax, nside=nside, cl=cl, nl_res=nl_res, eps_min=eps_min, ellscale=ellscale, g=g, tf=tf
        )
        self.set_ninv_model(ninv=ninv, ninv_nl=ninv_nl)
        self.s_inv_filt = hp_utils.jit(SkyInverseFilterJoint, s_cls=self.dl, nl_res=self.nl_res, tf=self.tf)

    @property
    def pre_op(self):
        if self._pre_op is None:
            self._pre_op = opfilt_hp.PreOperatorDiagJoint(
                self.s_inv_filt.s_cls, self.n_inv_filt, tf=self.tf, nl_res=self.s_inv_filt.nl_res
            )
        return self._pre_op

    def apply_ivf(self, tqumap, soltn=None):
        assert len(tqumap) == 3
        if soltn is not None:
            raise NotImplementedError
        else:
            teblm = np.zeros((3, hp.Alm.getsize(self.lmax)), dtype=np.complex128)
        self.solve(teblm, [tqumap[0], tqumap[1], tqumap[2]])
        # noinspection PyTypeChecker
        hp.almxfl(teblm[0], self.rescal_cl, inplace=True)
        # noinspection PyTypeChecker
        hp.almxfl(teblm[1], self.rescal_cl, inplace=True)
        # noinspection PyTypeChecker
        hp.almxfl(teblm[2], self.rescal_cl, inplace=True)
        return teblm

    def get_fl(self, pol, lmax):
        if pol == 'te':
            out = self.pre_op.fl[3, : self.lmax + 1] * self.rescal_cl[: self.lmax + 1] ** 2
        else:
            assert pol.lower() in 'teb'
            i = 'teb'.index(pol.lower())
            out = self.pre_op.fl[i, : self.lmax + 1] * self.rescal_cl[: self.lmax + 1] ** 2
        if lmax is None:
            return out
        else:
            return out[: lmax + 1]


class library_cinv_sTP:
    """Library to perform inverse-variance filtering of a simulation library.

    Suitable for separate temperature and polarization filtering.

    Parameters
    ----------
    sim_lib:
        simulation library instance (requires get_sim_tmap, get_sim_pmap methods)
    cinvt:
        temperature-only filtering library
    cinvp:
        poalrization-only filtering library
    """

    def __init__(self, sim_lib, cinvt: cinv_t, cinvp: cinv_p, lfilt=None, add_noise=False):
        self.sim_lib = sim_lib
        self.lfilt = lfilt
        self.add_noise = add_noise
        self.cinv_t = cinvt
        self.cinv_p = cinvp
        self.g = cinvp.g
        self.lmax = self.cinv_t.lmax if self.cinv_t is not None else self.cinv_p.lmax

    def get_fl(self, pol, lmax=None):
        if pol == 'te':
            out = np.zeros(self.lmax + 1)  # sep filtering/SQE doesn't need TE.
            if lmax is not None:
                return out[: lmax + 1]
            return out
        elif pol == 't':
            return self.cinv_t.get_fl(pol='t', lmax=lmax)
        elif pol in ['e', 'b']:
            return self.cinv_p.get_fl(pol=pol, lmax=lmax)
        else:
            raise ValueError("pol must be 't'/'e'/'b'/'te'")

    def _apply_ivf_t(self, tmap, soltn=None):
        return self.cinv_t.apply_ivf(tmap, soltn=soltn)

    def _apply_ivf_p(self, pmap, soltn=None):
        return self.cinv_p.apply_ivf(pmap, soltn=soltn)

    def get_sim_tlm(self, seed, cmbset, bundle):
        """
        Returns an inverse-filtered temperature simulation.

        Parameters
        ----------
        seed: int
            simulation index
        cmbset: str
             simulation set a/b
        bundle: int
            bundle index

        Returns
        -------
        tlm: np.ndarray
            inverse-filtered temperature healpy alm array
        """
        soltn = None
        map_in = self.sim_lib.get_tmap(seed, cmbset, bundle=bundle, add_noise=self.add_noise, g=self.g)
        tlm = self._apply_ivf_t(map_in, soltn=soltn)
        if self.lfilt is not None:
            # noinspection PyTypeChecker
            hp.almxfl(tlm, self.lfilt, inplace=True)
        return tlm

    def get_sim_eblm(self, seed, cmbset, bundle):
        """Returns an inverse-filtered E-polarization simulation.

        Parameters
        ----------
        seed: int
            simulation index
        cmbset: str
             simulation set a/b
        bundle: int
            bundle index

        Returns
        -------
        elm, blm
            inverse-filtered E/B alm arrays
        """
        map_in = self.sim_lib.get_pmap(seed, cmbset, bundle=bundle, add_noise=self.add_noise, g=self.g)
        elm, blm = self._apply_ivf_p(map_in, soltn=None)

        if self.lfilt is not None:
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)
        return elm, blm

    def get_eps(self):
        out = []
        if self.cinv_t is not None:
            out += list(self.cinv_t.eps)
        if self.cinv_p is not None:
            out += list(self.cinv_p.eps)
        return np.array(out)


class library_cinv_jTP:
    """Library to perform inverse-variance filtering of a simulation library.

    Suitable for joint temperature and polarization filtering.

    Parameters
    ----------
    sim_lib:
        simulation library instance (requires get_sim_tmap, get_sim_pmap methods)
    cinv_jtp:
        temperature and pol joint filtering library
    """

    def __init__(self, sim_lib, cinv_jtp: cinv_tp, lfilt=None, add_noise=False):
        self.sim_lib = sim_lib
        self.lfilt = lfilt
        self.add_noise = add_noise
        self.cinv_tp = cinv_jtp
        self.g = cinv_jtp.g

    def get_sim_teblm(self, seed, cmbset, bundle):
        return self._get_alms("teb", seed, cmbset, bundle=bundle)

    def _get_alms(self, a, seed, cmbset, bundle):
        assert a in ["t", "e", "b", "teb"]

        T = self.sim_lib.get_tmap(seed, cmbset, bundle=bundle, add_noise=self.add_noise, g=self.g)
        Q, U = self.sim_lib.get_pmap(seed, cmbset, bundle=bundle, add_noise=self.add_noise, g=self.g)
        tlm, elm, blm = self._apply_ivf([T, Q, U], soltn=None)

        if self.lfilt is not None:
            hp.almxfl(tlm, self.lfilt, inplace=True)
            hp.almxfl(elm, self.lfilt, inplace=True)
            hp.almxfl(blm, self.lfilt, inplace=True)

        return {"teb": (tlm, elm, blm), "t": tlm, "e": elm, "b": blm}[a]

    def _apply_ivf(self, tqumap, soltn=None):
        return self.cinv_tp.apply_ivf(tqumap, soltn=soltn)

    def get_fl(self, pol, lmax=None):
        return self.cinv_tp.get_fl(pol=pol, lmax=lmax)

    def get_eps(self):
        out = []
        if self.cinv_tp is not None:
            out += list(self.cinv_tp.eps)
        return np.array(out)


class MapsBase(abc.ABC):
    """Base class for maps, used for testing and as a base for Maps class."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_tmap(self, seed, cmbset, bundle, add_noise=True, apply_tf=False, g=None):
        """Get T map for the given seed and cmbset.

        Parameters
        ----------
        seed: int
            Seed for the map. "0" is reserved for data maps.
        cmbset: str
            cmbset of the sims. Available sets are single letter strings, e.g., 'a/b'.
        bundle: int
            integer number from 0--`config.nbundle`-1.
        add_noise: bool=False
            For simulations, this controls whether to add noise to the map.
        apply_tf: bool=False
            Whether to apply transfer function to the signal map.
        g: Geometry=None
            If provided, use the ducc_sht.Geometry to speed up SHT operations (recommended).

        Returns
        -------
        np.ndarray
            T map for the given seed and cmbset, shape (npix, ).
        """
        pass

    @abc.abstractmethod
    def get_pmap(self, seed, cmbset, bundle, add_noise=True, apply_tf=False, g=None):
        """Get Q/U map for the given seed and cmbset.

        Parameters
        ----------
        seed: int
            Seed for the map. "0" is reserved for data maps.
        cmbset: str
            cmbset of the sims. Available sets are single letter strings, e.g., 'a/b'.
        bundle: int
            integer number from 0--`config.nbundle`-1.
        add_noise: bool=False
            For simulations, this controls whether to add noise to the map.
        apply_tf: bool=False
            Whether to apply transfer function to the signal map.
        g: Geometry=None
            If provided, use the ducc_sht.Geometry to speed up SHT operations (recommended).

        Returns
        -------
        np.ndarray
            Q/U map for the given seed and cmbset, shape (2, npix).
        """
        pass

    @abc.abstractmethod
    def get_nlres(self, cinv=False):
        raise NotImplementedError("get_nlres method must be implemented in subclass")

    def load_sim_alm(self, config, seed, cmbset, bundle, add_noise):
        """Prepare the input alms for naive cinv-filtering."""
        t = self.get_tmap(cmbset=cmbset, seed=seed, bundle=bundle, add_noise=add_noise, g=config.g)
        q, u = self.get_pmap(cmbset=cmbset, seed=seed, bundle=bundle, add_noise=add_noise, g=config.g)
        t *= config.mask_cinv['t']
        q *= config.mask_cinv['p']
        u *= config.mask_cinv['p']
        maps = np.array([t, q, u])
        alm = config.g.map2alm(maps, lmax=config.cinv_lmax, check=False).astype(np.complex128)
        if config.tf2d:
            alm[0] *= cinv_utils.cli(config.tfbl_2d('t'))
            alm[1] *= cinv_utils.cli(config.tfbl_2d('p'))
            alm[2] *= cinv_utils.cli(config.tfbl_2d('p'))
        else:
            hp.almxfl(alm[0], cinv_utils.cli(config.tfbl_1d('t')), inplace=True)
            hp.almxfl(alm[1], cinv_utils.cli(config.tfbl_1d('p')), inplace=True)
            hp.almxfl(alm[2], cinv_utils.cli(config.tfbl_1d('p')), inplace=True)
        return alm

    def naive_cinv(self, config, seed, cmbset, bundle, add_noise):
        """Return naively cinv-filtered alms (and fls) as arrays."""
        from healqest.ducc_sht import reduce_lmax

        nlres = self.get_nlres(cinv=False)
        alms = self.load_sim_alm(config, seed=seed, cmbset=cmbset, bundle=bundle, add_noise=add_noise)
        almbars = np.zeros((3, hp.Alm.getsize(config.lmax)), dtype=np.complex128)
        flms = np.zeros((4, config.lmax + 1)).astype(float)
        cls = config.cinv_cls['cmb']

        for j, s in enumerate('teb'):
            ss = f"{s.lower()}{s.lower()}"
            assert ss in ['tt', 'ee', 'bb']
            lmax = config.lmaxT if s == 't' else config.lmaxP
            lmin = config.lminT if s == 't' else config.lminP
            fl = cinv_utils.cli(cls[ss][: config.lmax + 1] + nlres[ss][: config.lmax + 1])
            fl[:lmin] = 0
            fl[lmax + 1 :] = 0
            # if alms is None:
            #     return fl
            alm = hp.almxfl(reduce_lmax(alms[j], config.lmax), fl)
            almbars[j], flms[j] = alm, fl
        return almbars, flms


class ILCData:
    """Class to hold ILC/cinv related data.

    Notes
    -----
    The ILC data is fully parsed from a pre-generated ILC file. So the attributes of this class should all be
    implemented in the `.npz` file from which the `ILCData` is instantiated.

    Attributes
    ----------
    bl_sim: dict
        dict of simulation beams with 90/150/220/effective keys. Each value is an array of sufficient length.
    bl_t, bl_p: dict
        dict of data temperature/polarization beams with 90/150/220 keys. Each value is an array of sufficient
        length.
    weights: np.ndarray
        Array of shape (2, 3, lmax). The per-ell ILC weights For T and P, 3 frequencies.
    ilc_nlres: np.ndarray
        Array of shape (2, lmax+1). The ILC noise+residual (subtracting the white noise part)
    ilc_ninv: np.ndarray
        Array of shape (2, npix). The ILC noise inverse in pixel space. This is built from the sparse `ninv`
        array from the ILC file, with `nside` and `ninv_ipix` infos.
    nep: np.ndarray
        Array of shape (2,). The white noise equivalent temperature for T and P.
    ninv_nl: np.ndarray
        Array of shape (2, lmax+1). The harmonic space ninv_nl for `NoiseInverseFilterAlm`

    """

    def __init__(self, fname, ilc_type):
        """
        Initialize the ILCData object by loading the ILC file and parsing the relevant data.

        Parameters
        ----------
        fname: str
            The file name of the ILC data file, which should be a `.npz`
        ilc_type: str
            ilc type, e.g., 'mv', 'tszfree', 'cibfree'.
        """
        loaded = np.load(fname, allow_pickle=True)
        self.bl_sim = loaded['bl_sim'].item()
        self.bl = self.bl_sim['effective']
        self.bl_t = loaded['bl_t'].item()
        self.bl_p = loaded['bl_p'].item()
        self.weights = loaded['weights'].item()[ilc_type]
        self.ilc_nlres = loaded['ilc_nlres'].item()[ilc_type]
        self.ilc_ninv = np.zeros((2, hp.nside2npix(loaded['nside'])))
        ipix = loaded['ninv_ipix']
        self.ilc_ninv[:, ipix] = loaded['ninv'].item()[ilc_type]
        self.lmax = self.weights.shape[-1] - 1

        # nep is optional. For inhomogeneous alm model, this is not needed.
        if 'nep' in loaded.files:
            self.nep = loaded['nep'].item()[ilc_type]
        else:
            self.nep = None

        if 'ninv_nl' in loaded.files:
            self.ninv_nl = loaded['ninv_nl'].item()[ilc_type]
        else:
            self.ninv_nl = None
