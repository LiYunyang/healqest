"""
This module provides setups needed at startup, including
    - parsing config files;
    - initiating the global logger
    - modules for map io (might move maps sometime).
"""
import argparse
from functools import cached_property
from importlib import resources
import logging
import os
import string
import shutil
import sys
from typing import Union, get_type_hints
import warnings

import numpy as np
import healpy as hp
import yaml
from git import Repo, InvalidGitRepositoryError

from healqest import utils, healqest_utils
from healqest.ducc_sht import Geometry
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
        warnings.filterwarnings("ignore")
except ImportError:
    rank = 0

logger = logging.getLogger(__name__)


def get_git_version():
    """Get the current git commit hash of the repository."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        repo = Repo(repo_dir, search_parent_directories=True)
        commit_hash = repo.head.commit.hexsha[:7]  # Short hash
        dirty = repo.is_dirty()
        return commit_hash + ("-dirty" if dirty else "")
    except InvalidGitRepositoryError:
        return "unknown"


class PartialFormatter(string.Formatter):
    """Allow delayed fornatting of values in strings."""
    def get_value(self, key, args, kwargs):
        # Try to resolve the key
        try:
            return super().get_value(key, args, kwargs)
        except (KeyError, IndexError):
            # Preserve the original placeholder if value is missing
            if isinstance(key, str):
                return "{" + key + "}"
            return super().get_value(key, args, kwargs)

    def format_field(self, value, format_spec):
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            v = value[1:-1]
            # If value is a placeholder, return it with its format spec untouched
            return f"{{{v}:{format_spec}}}" if format_spec else value
        try:
            return super().format_field(value, format_spec)
        except (ValueError, TypeError):
            return f"{value}:{format_spec}"


class Config:
    __keywords__ = ['base', 'lensrec', 'inputs', 'pspec', 'cinv']

    # === command-line parameters ===
    field: str = None  # SPT field name, used for output/fname parsing and mask selection

    """Config should be specified in a yaml file with these keywords. """
    # === base ===
    outdir: str
    recdir: str = None  # output directory for lensing rec. Default to outdir
    dec_range: Union[list, dict] = None
    save_as_map: bool = False  # save plm as map, otherwise as alm.
    nbundle: int = None  # number of bundles, if any.

    # === cinv ===
    eps_t: float  # convergence threshold for cinv T component
    eps_p: float  # convergence threshold for cinv Pol component
    cinv_lmax: int  # maximum l for cinv
    cinv_lmin: int  # minimum l for cinv
    file_bl: str  # path to beam file.
    file_tf2d: str=None  # path to tf2d file
    file_cambcmb: str  # path to the camb cls file for cinv (relative to healqest/camb)
    file_noisefg: str  # path to the noise + foreground (tf2d+beam-ed)
    file_slm: str  # path to (beamed) signal alm files for std/N0-type sims.
    file_nlm: str = None  # path to noise alm files for std/N0-type sims.
    file_slm_N1: str  # path to (beamed) signal alm files for N1-type sims.
    file_nlm_N1: str = None  # path to noise alm files for N1-type sims.
    nlev_t: float = None  # NET value, if not specified, nlev_t = nlev_p / sqrt(2)
    nlev_p: float = None  # NEQ/U values, if not specified, nlev_p = nlev_t * sqrt(2)
    ellscale: bool = True  # if True, apply the l(l+1)/2pi scaling to cinv cls
    cinvdir: str = None # The output directory for cinv maps. If not specified, set to "recdir"

    # === lensrec ===
    rectype: str  # [sqe,gmv,mh,xilc,gmvph]
    mvtypes: list[str]  # subset of ['TT', 'TE', 'EE', 'TB', 'EB'] for SQE
    nside: int  # Map nside, used for ducc wrapper. This is enforced for lensrec too, so 2nside>Lmax
    lminT: int  # override lmin
    lminP: int  # override lmin
    lmin: int = None  # default lmin if lminT/P is not specified.
    lmaxT: int  # override lmax
    lmaxP: int  # override lmax
    lmax: int = None  # default lmax if lmaxT/P is not specified
    Lmax: int  # maximum reconstruction L
    file_cmb: str  # Path to cmb spectrum used for lensrec
    fmask_qe: Union[str, list[str]] = None  # path(s) to mask used for lensrec

    # === inputs ===
    sim_range: list[int]  # index range (inclusive) of the input alms files
    file_alm: Union[str, list]  # path or list of path to input alms files
    file_alm_N1: Union[str, list]  # path or list of path to input alms files for N1
    kappa_in: str  # fname of input kappa maps
    clkk_in: str  # fname of input kappa spectrum

    # === pspec ===
    mfsplit: bool = True  # split mean-field sims by halves for auto spectra
    fmask_ps: Union[str, list[str]] = None  # path(s) to mask used for clpp
    fmask_resp: Union[str, list[str]] = None  # path(s) to mask used for MC resp
    spice_kwargs: dict = None  # polspice settings

    def __init__(self, cinv=None, **kwargs):
        self._validate_config(kwargs)
        self.__dict__.update(kwargs)
        self._set_defaults()

    @classmethod
    def from_yaml(cls, fname, field=None):
        params = yaml.safe_load(open(fname, "r"))
        config_dict = dict(field=field)
        for group_key in cls.__keywords__:
            subdict = params.pop(group_key, None)
            if subdict:
                config_dict.update(**subdict)

        for key in params.keys():
            warnings.warn(f"Ungrouped config parameter: '{key}'. ", UserWarning, stacklevel=2)
        config_dict.update(**params)  # TODO: this is for backward compatibility
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args):
        fname = args.config
        obj = cls.from_yaml(fname, field=args.field)
        if rank == 0:
            os.makedirs(obj.path(obj.outdir), exist_ok=True)
            for i, f in enumerate([fname, sys._getframe(1).f_globals['__file__']]):
                if i > 0:
                    # add git hash for scripts, not the config file.
                    name, ext = os.path.splitext(os.path.basename(f))
                    out_fname = f"{name}.{get_git_version()}{ext}"
                else:
                    out_fname = os.path.basename(f)
                shutil.copy(f, obj.path(obj.outdir, out_fname))
        return obj

    def _validate_config(self, config_dict: dict):
        expected_keys = get_type_hints(self)
        unexpected_keys = set(config_dict.keys()) - set(expected_keys.keys())
        for key in unexpected_keys:
            warnings.warn(f"Unexpected config parameter: '{key}'. ", UserWarning, stacklevel=2)

    def _set_defaults(self):
        # set default lmax/lmax
        self.lmaxT = getattr(self, 'lmaxT', self.lmax)
        self.lmaxP = getattr(self, 'lmaxP', self.lmax)
        self.lminT = getattr(self, 'lminT', self.lmin)
        self.lminP = getattr(self, 'lminP', self.lmin)
        self.lmax = max(self.lmaxT, self.lmaxP)
        self.lmin = min(self.lminT, self.lminP)
        for key in ['lmaxT', 'lmaxP', 'lminT', 'lminP']:
            assert getattr(self, key) is not None

        # set default paths
        if self.recdir is None:
            self.recdir = self.outdir
        if self.cinvdir is None:
            self.cinvdir = self.recdir
        if self.field is not None:
            self.outdir = self.outdir.format(field=self.field)
            self.recdir = self.recdir.format(field=self.field)
            self.cinvdir = self.cinvdir.format(field=self.field)

        # set field specific settings
        if isinstance(self.dec_range, dict):
            if self.field is not None:
                self.dec_range = self.dec_range[self.field]
        # auto adjust spice kwargs
        if self.spice_kwargs:
            for key in ['apodizesigma', 'thetamax']:
                if self.spice_kwargs.get(key, None) == 'dec':
                    self.spice_kwargs[key] = max(self.dec_range) - min(self.dec_range)
                elif isinstance(self.spice_kwargs[key], dict):
                    self.spice_kwargs[key] = self.spice_kwargs[key][self.field]

        # cinv settings:
        if self.nlev_p is None and self.nlev_t is not None:
            self.nlev_p = self.nlev_t * np.sqrt(2)
        if self.nlev_t is None and self.nlev_p is not None:
            self.nlev_t = self.nlev_p / np.sqrt(2)

        # mvtypes has to be a list
        self.mvtypes = list(self.mvtypes)

    @staticmethod
    def path(path, *args, **kwargs):
        """
        Formatting the dir/file name.

        Parameters
        ----------
        path : str
            The path to the file. If it starts with "/", it is considered an absolute path.
            Otherwise, it is considered relative to the environ-specific $HEALQEST_IO_ROOT.
        args: str
            Additional path components to be joined to the path.
        kwargs:
            Dictionary to be used for formatting the path.
        """
        if path is None:
            return None
        if path.startswith('/'):
            out = path
        else:
            out = os.path.join(os.environ["HEALQEST_IO_ROOT"], path)
        if args:
            out = os.path.join(out, *args)
        if kwargs:
            out = out.format(**{k: v for k, v in kwargs.items() if v is not None})
        if os.path.exists(out):
            pass
        else:
            pass
        return out

    @staticmethod
    def as_list(x):
        if isinstance(x, list):
            return x
        elif isinstance(x, str):
            return [x]
        else:
            raise NotImplementedError(f"turning {type(x)} into list is not implemented")

    @staticmethod
    def ktype2ij(ktype, i, j=None) -> (int, int, str, str):
        """
        Convert the 2-letter ktype to seed and cmbset of the two maps
        """
        if j is None and len(set(ktype)) == 2:
            # default `seed2` is `seed1 + 1`
            j = i+1
        if ktype in ['xx', 'xy', 'yx', 'x0', '0x']:
            cmbset1, cmbset2 = 'aa'
            if ktype == 'xx':
                seed1, seed2 = i, i
            elif ktype == 'xy':
                seed1, seed2 = i, j
            elif ktype == 'yx':
                seed1, seed2 = j, i
            elif ktype == 'x0':
                seed1, seed2 = i, 0
            elif ktype == '0x':
                seed1, seed2 = 0, i
            else:
                raise AssertionError
        elif ktype in ['ab', 'ba']:
            cmbset1, cmbset2 = ktype
            seed1, seed2 = i, i
        else:
            raise TypeError(f'Undefined ktype {ktype}')
        return seed1, seed2, cmbset1, cmbset2

    @staticmethod
    def mvtype2qe(mvtype):
        # SQE type MVs
        if mvtype in ['MV', 'qMV', 'PP', 'qPP', 'TTEETE', 'qTTEETE']:
            return healqest_utils.get_qes(mvtype)
        elif mvtype in ['TT', 'TE', 'TB', 'EE', 'EB', 'ET', 'BT', 'BE']:
            return healqest_utils.get_qes(mvtype)
        elif mvtype == 'GMV':
            return ['TT_GMV', 'EE_GMV', 'TE_GMV', 'ET_GMV', 'TB_GMV', 'BT_GMV', 'EB_GMV', 'BE_GMV']
        else:
            raise ValueError(f'Undefined mvtype: {mvtype}')

    @property
    def qes(self):
        """return qes for lensrec"""
        qes = list()
        for mvtype in self.mvtypes:
            qes += self.mvtype2qe(mvtype)
        return list(set(qes))

    @cached_property
    def cinv_cls(self) -> dict:
        """the CMB and beam-convolved foreground cls for cinv

        Note
        ----
        The lmax is set to `cinv_lmax`, which should be large than any other possible
        lmax used elsewhere in the pipeline.
        """
        out = dict()

        def dat2dict(ell, dat):
            assert ell[0] == 0
            assert ell[-1] == self.cinv_lmax
            return dict(tt=dat[0],
                        ee=dat[1],
                        bb=dat[2],
                        te=dat[3])

        file_cmb = str(resources.files('healqest') / 'camb' / self.file_cambcmb)
        ell, *dat = utils.get_lensedcls(file_cmb, lmax=self.cinv_lmax)
        out['cmb'] = dat2dict(ell, dat)

        if hasattr(self, 'file_noisefg'):
            ell, *dat = np.loadtxt(self.path(self.file_noisefg))[:self.cinv_lmax + 1, :5].T
            out['nl_res'] = dat2dict(ell, dat)
        else:
            out['nl_res'] = None
        return out

    @cached_property
    def cmbcl(self):
        cambcls = str(resources.files('healqest') / 'camb' / self.file_cmb)
        ell, sltt, slee, slbb, slte = utils.get_lensedcls(cambcls, lmax=self.lmax)
        return dict(tt=sltt, ee=slee, bb=slbb, te=slte)

    @cached_property
    def tfbl_1d(self) -> dict:
        """Return 1d bl functions for T/E/B with lmax=`cinv_lmax`"""
        if self.file_bl is None:
            logger.error("beam file not given! assuming unity bl for now.")
            bl = np.ones(self.cinv_lmax)
        else:
            beam_file = self.path(self.file_bl)
            logger.warning("temporarily loading the 150 GHz beam file")
            bl = np.loadtxt(beam_file)[:self.cinv_lmax + 1, 2]  # TODO: default to 150 GHz for the test file
        return dict(t=bl, e=bl, b=bl)

    @cached_property
    def tfbl_2d(self) -> Union[dict, None]:
        """Return a 2d bl/tf functions for T/E/B"""
        if self.file_tf2d:
            raise NotImplementedError
            return np.load(self.file_tf2d)  # TODO: actually implement this when needed
        else:
            logger.warning("No 2d TF given.")
            return None

    @cached_property
    def g(self):
        return Geometry(nside=self.nside, dec_range=getattr(self, 'dec_range', None))

    # === setup masks ===
    def _load_mask(self, item):
        mask = None
        for _ in self.as_list(item):
            _mask = hp.read_map(self.path(_, field=self.field))
            _mask[hp.mask_bad(_mask)] = 0
            if mask is None:
                mask = _mask
            else:
                mask *= _mask
        return mask

    @cached_property
    def mask_qe(self):
        if self.fmask_qe:
            return self._load_mask(self.fmask_qe)
        else:
            return np.ones(hp.nside2npix(self.nside))

    @cached_property
    def mask_ps(self):
        if self.fmask_ps:
            return self._load_mask(self.fmask_ps)
        else:
            return self.mask_qe

    @cached_property
    def mask_resp(self):
        if self.fmask_resp:
            return self._load_mask(self.fmask_resp)
        else:
            return self.mask_ps

    @cached_property
    def mask_boundary(self):
        """boundary mask used to save plm as partial maps"""
        return self.mask_qe !=0

    # === setup paths ===
    def p_cinv(self, cinv_type:str, seed, cmbset, N1=False, bundle=None, suffix='fits'):
        """
        Path to the cinv files (as output of cinv, or input of reclens)

        Parameter
        ---------
        cinv_type: str
            Type of the cinv file. Can be stp or jtp
        seed: int
        cmbset: str
            Single letter strings. Accepted values are 'a', 'b'
        N1: bool=False
        bundle: int=None
        """

        N1_tag = "_N1" if N1 else ""
        subdir = 'cinv'
        if bundle is not None:  # MF and N1 don't do bundle
            subdir = f"{subdir}/bundle{bundle}"
        fname = f'cinv_{cinv_type}_{seed}_{cmbset}{N1_tag}.{suffix}'
        return self.path(self.cinvdir, subdir, fname)

    def p_plm(self, tag=None, seed1=None, seed2=None, cmbset1=None, cmbset2=None, N1=False, stack_type=None,
              bundle=None):
        """
        Return paths to plm(stacked) files.

        Parameters
        ----------
        tag: str
            name of a QE or a MVtype.
        seed1, seed2: int
        cmbset1, cmbset2: str
            Single letter strings. Accepted values are 'a', 'b', 'x', 'y'.
        N1: bool=False
            Indicate if the target file is for N1 calculation (they live in a separate directory).
        stack_type: str
            Indicate the stacking type for mean-field calculations.
        bundle: int=None
        """
        subdir = 'lensrec_N1' if N1 else 'lensrec'
        if bundle is not None and not N1:  # MF and N1 don't do bundle
            subdir = f"{subdir}/bundle{bundle}"
        suffix = 'fits' if self.save_as_map else 'npz'
        if not stack_type:
            if self.save_as_map:
                # when saving as partial maps, save all QEs together as columns, so QE doesn't appear in fname.
                fname = f'plm_{seed1}{cmbset1}_{seed2}{cmbset2}.{suffix}'
            else:
                fname = f'plm_{tag}_{seed1}{cmbset1}_{seed2}{cmbset2}.{suffix}'
        else:
            subdir = f"{subdir}/stack"
            fname = f'plmstack_{tag}_{stack_type}.{suffix}'
        out = self.path(self.recdir, subdir, fname)
        return out

    def p_cls(self, tag, seed1, seed2, ktype1, ktype2, N1=False, ext='dat'):
        """paths to power spectra files."""

        subdir = 'cls/lensrec_N1' if N1 else 'cls'
        s1, s2, c1, c2 = self.ktype2ij(ktype1, seed1, seed2)
        tag1 = f"{s1}{c1}_{s2}{c2}"
        if ktype2 is not None:
            assert set(ktype1) == set(ktype2)
            s1, s2, c1, c2 = self.ktype2ij(ktype2, seed1, seed2)
            tag2 = f"{s1}{c1}_{s2}{c2}"
            fname = f'clkk_k{tag}_{tag1}_{tag2}.{ext}'
        else:
            fname = f'clkk_k{tag}_{tag1}_cross.{ext}'
        out = self.path(self.outdir, subdir, fname)
        return out

    def p_resp(self, tag, bundle=None):
        """paths to response functions."""
        bundle_tag = f'_bundle{bundle}' if bundle is not None else ''
        return self.path(self.outdir, f"respavg_{tag}{bundle_tag}.npz")

    @staticmethod
    def f_tmp(tag, seed1=None, seed2=None, ktype=None, N1=False, mf_group=0, bundle=None):
        """
        Return file name of a temprary file for kappa maps.

        Parameters
        ----------
        tag: str
            qe or a mvtype, e.g. "TT"/"qMV"/"GMV"/"PP"
        seed1, seed2: int
        ktype: str
            2-letter string
        N1: bool=False
            Indicator for N1-type maps.
        mf_group: int=0
        bundle: int=None
        """

        bundle_tag = f'_bundle{bundle}' if bundle is not None else ''
        N1_tag = '_N1' if N1 else ''
        return os.path.join(f"kmap_{tag}{bundle_tag}_{seed1}_{seed2}_mfgroup{mf_group}_{ktype}{N1_tag}.tmp")


class Maps:
    def __init__(self, nside, file_signal, file_noise=None, tf2d=None, config=None, N1=False, bundle=None):
        """
        Maps class for loading maps as simulation/cinv input

        Parameters
        ----------
        nside: int
            Nside for the maps
        file_signal: str
            String or format string that expects {seed} and {cmbid} substitutions. This points to signal alm files.
        file_noise: str=None
            String or format string that expects {seed} and {cmbid} substitutions. This points to noise alm files.
        tf2d: np.array=None
            2d TF in alm space
        config: Config=None
        N1: bool=False
            Specify if these maps are for N1-type or std sims. This is only relevant for noise seed generations if
            `file_noise` is not given.
        bundle: int=None
            Specify if the bundle. This is only relevant for noise seed generations if `file_noise` is not given.
        """
        self.nside = nside
        self.file_cmb = file_signal
        self.file_noise = file_noise
        self.tf2d = tf2d
        self.config = config
        self.bundle = bundle
        self.N1 = N1

    @classmethod
    def from_config(cls, config: Config, N1=False, bundle=None):
        if not N1:
            _file_signal, _file_noise = config.file_slm, config.file_nlm
        else:
            _file_signal, _file_noise = config.file_slm_N1, config.file_nlm_N1

        # Apply additional bundle/field specifications with PartialFormatter
        f = PartialFormatter()
        file_signal = f.format(config.path(_file_signal), field=config.field, bundle=bundle)
        if _file_noise is not None:
            file_noise = f.format(config.path(_file_noise), field=config.field, bundle=bundle)
        else:
            file_noise = None

        return cls(
            nside=config.nside,
            file_signal=file_signal,
            file_noise=file_noise,
            config=config,
            tf2d=config.tfbl_2d,
            N1=N1, bundle=bundle
        )

    def get_tmap(self, seed, cmbid, add_noise=True, apply_tf=False, g=None):
        """Load sim temperature signal and noise map separately and add
        """

        f_slm = self.file_cmb.format(seed=seed, cmbid=cmbid)
        logger.debug(f"loading signal sim from {f_slm}")
        almt = hp.read_alm(f_slm, hdu=1)
        lmax_in = hp.Alm.getlmax(almt.shape[-1])
        if add_noise:
            if self.file_noise is not None:
                f_nlm = self.file_noise.format(seed=seed, cmbid=cmbid)
                logger.info(f"Adding noise: {f_nlm}")
                nlm = hp.read_alm(f_nlm, hdu=1)
            else:
                nlm = self.add_noise_t(seed=seed, cmbid=cmbid, bundle=self.bundle, N1=self.N1)
                nlm = utils.reduce_lmax(nlm, lmax=lmax_in)
            almt += nlm
            del nlm
        else:
            pass

        if apply_tf:
            logger.debug("Applying tf")
            assert self.tf2d is not None
            lmax = hp.Alm.getlmax(len(self.tf2d))
            if lmax_in > lmax:
                almt = utils.reduce_lmax(almt, lmax)
            almt *= self.tf2d
        else:
            pass
        if g is None:
            logger.warning("using healpy alm2map")
            return hp.alm2map(almt, nside=self.nside)
        else:
            return g.alm2map(almt)

    def get_pmap(self, seed, cmbid, add_noise=True, apply_tf=False, g=None):
        """Load sim E/B signal and noise map separately and add"""

        f_slm = self.file_cmb.format(seed=seed, cmbid=cmbid)
        logger.debug(f"loading signal sim from {f_slm}")
        almeb = hp.read_alm(f_slm, hdu=[2, 3])
        lmax_in = hp.Alm.getlmax(almeb.shape[-1])

        if add_noise:
            if self.file_noise is not None:
                f_nlm = self.file_noise.format(seed=seed, cmbid=cmbid)
                logger.info(f"Adding noise: {f_nlm}")
                nlm = hp.read_alm(f_nlm, hdu=[2, 3])
            else:
                nlm = self.add_noise_p(seed=seed, cmbid=cmbid, bundle=self.bundle, N1=self.N1)
                nlm = utils.reduce_lmax(nlm, lmax=lmax_in)
            almeb += nlm
            del nlm
        else:
            pass

        if apply_tf:
            logger.debug("Applying tf")
            assert self.tf2d is not None
            lmax = hp.Alm.getlmax(len(self.tf2d))
            if lmax_in > lmax:
                almeb = utils.reduce_lmax(almeb, lmax)
            almeb *= self.tf2d
        else:
            pass
        if g is None:
            logger.warning("using healpy alm2map")
            return hp.alm2map_spin(almeb, nside=self.nside, spin=2, lmax=hp.Alm.getlmax(almeb.shape[-1]))
        else:
            return g.alm2map(almeb)

    @staticmethod
    def add_noise_t(seed, cmbid, bundle, N1) -> np.ndarray:
        pass

    @staticmethod
    def add_noise_p(seed, cmbid, bundle, N1) -> np.ndarray:
        pass


def parser():
    p = argparse.ArgumentParser()

    p.add_argument('-c', '--config', default=None, type=str, help='path to config file', required=True)
    p.add_argument('-f', '--field', default=None, type=str, help='SPT field')

    # bundle and n1 are mutually exclusive: N1-type calculations should be the same for all bundles.
    group = p.add_mutually_exclusive_group()
    group.add_argument('-b', '--bundle', type=int, help='Bundle id')
    group.add_argument('-n1', action='store_true', help='Do N1-type operations')

    p.add_argument('-rdn0', action='store_true', help='do RDN0-type operations')
    p.add_argument('-skip', action='store_true', help='skip finished jobs')
    p.add_argument('-v', '--verbose', default=3, type=int, help='Output verbosity')
    return p


class MPIAwareFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",  # white
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    RESET = "\033[0m"
    FAINT = "\033[2;37m"

    def format2(self, record):
        record.rank = rank  # Save rank into the record
        record.name = record.name.split('.')[-1]  # truncate to keep only the module name
        record.name = f"\033[2;37m{record.name}{self.RESET}"
        asctime = self.formatTime(record, self.datefmt)
        color = self.COLORS.get(record.levelno, "")
        prefix = f"[{rank}]{color}{asctime}{self.RESET}|"

        # Format rest of the message
        message = super().format(record)

        # Replace the [asctime(rank)] part with the colored version
        # This assumes `[{asctime}({rank})]` is at the beginning of your format string
        end = message.find("|") + 1
        return f"{prefix}{message[end:]}"

    def format(self, record):
        # Fixed-width module name
        name = record.name.split(".")[-1][:10].rjust(10)
        name_colored = f"{self.FAINT}{name}{self.RESET}"

        # Faint rank (global rank must be defined elsewhere)
        rank_colored = f"{self.FAINT}{rank}{self.RESET}"

        # Time colored by log level
        asctime = self.formatTime(record, self.datefmt)
        time_color = self.COLORS.get(record.levelno, "")
        asctime_colored = f"{time_color}{asctime}{self.RESET}"

        # Actual log message
        message = record.getMessage()

        return f"{rank_colored}|{asctime_colored}|{name_colored} {message}"


def verbose2level(verbosity: int) -> int:
    verbose = int(verbosity)
    verbose = max(verbose, 0)
    verbose = min(verbose, 4)
    return {
        0: logging.CRITICAL,
        1: logging.ERROR,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG,
    }[verbose]


def setup_logger(verbose=3, force=True, quiet=True):
    """
    Central logging config. Call this early in your script.

    Parameters
    ----------
    verbose: int=3
        verboisity. 0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG
    force: bool=True
    quiet: bool=True
        If True, scilence sub-warning level message from selected modules.
    """
    if rank > 0:
        # Silence all logging on non-root ranks
        logging.disable(logging.CRITICAL)
        # return
        pass

    fmt = "[{rank}]{asctime}|{name} {message}"
    datefmt = "%H:%M:%S"

    formatter = MPIAwareFormatter(fmt, datefmt, style='{')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(level=verbose2level(verbose), handlers=[handler], force=force)

    if quiet:
        for name in logging.root.manager.loggerDict:
            if not name.startswith("healqest") and name != "__main__":
                logging.getLogger(name).setLevel(logging.WARNING)
            else:
                pass
