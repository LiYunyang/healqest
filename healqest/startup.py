import sys
import argparse
import os
from typing import Union, get_type_hints

import numpy as np
import yaml
from healqest import utils, healqest_utils
from importlib import resources
import healpy as hp
from healqest.ducc_sht import Geometry
from functools import cached_property
import shutil
import warnings
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
        warnings.filterwarnings("ignore")
except ImportError:
    rank = 0


class Config:
    __keywords__ = ['base', 'lensrec', 'inputs', 'pspec', 'cinv']

    # === command-line parameters ===
    field: str = None  # SPT field name, used for output/fname parsing and mask selection
    bundle: int = None  # bundle number, used for output/fname parsing

    """Config should be specified in a yaml file with these keywords. """
    # === base ===
    outdir: str
    recdir: str = None  # output directory for lensing rec. Default to outdir
    dec_range: Union[list, dict] = None
    save_as_map: bool = False  # save plm as map, otherwise as alm.

    # === cinv ===
    file_bl: str  # path to beam file.

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

    def __init__(self, **kwargs):
        self._validate_config(kwargs)
        self.__dict__.update(kwargs)
        self._set_defaults()

    @classmethod
    def from_yaml(cls, fname, field=None, bundle=None):
        params = yaml.safe_load(open(fname, "r"))
        config_dict = dict(field=field, bundle=bundle)
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
        obj = cls.from_yaml(fname, field=args.field, bundle=args.bundle)
        if rank == 0:
            os.makedirs(obj.path(obj.outdir), exist_ok=True)
            shutil.copy(fname, obj.path(obj.outdir))
            script = sys._getframe(1).f_globals['__file__']
            shutil.copy(script, obj.path(obj.outdir))
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
        if self.field is not None:
            self.outdir = self.outdir.format(field=self.field)
            self.recdir = self.recdir.format(field=self.field)

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

    @cached_property
    def cmbcl(self):
        cambcls = str(resources.files('healqest') / 'camb' / self.file_cmb)
        ell, sltt, slee, slbb, slte = utils.get_lensedcls(cambcls, lmax=self.lmax)
        return dict(tt=sltt, ee=slee, bb=slbb, te=slte)

    @property
    def qes(self):
        """return qes for lensrec"""
        qes = list()
        for mvtype in self.mvtypes:
            qes += self.mvtype2qe(mvtype)
        return list(set(qes))

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

    @cached_property
    def g(self):
        return Geometry(nside=self.nside, dec_range=getattr(self, 'dec_range', None))

    def _load_mask(self, item):
        if isinstance(item, str):
            return hp.read_map(self.path(item, field=self.field))
        elif isinstance(item, list):
            mask = None
            for _ in item:
                _mask = self._load_mask(_)
                if mask is None:
                    mask = _mask
                else:
                    mask *= _mask
            return mask
        else:
            raise TypeError(f'Undefined item type {type(item)}')

    @cached_property
    def mask_qe(self):
        if self.fmask_qe:
            return self._load_mask(self.fmask_qe)
        else:
            return None

    @cached_property
    def mask_ps(self):
        if self.fmask_ps:
            return self._load_mask(self.fmask_ps)
        else:
            return None

    @cached_property
    def mask_resp(self):
        if self.fmask_resp:
            return self._load_mask(self.fmask_resp)
        else:
            return None

    @cached_property
    def mask_bounary(self):
        """boundary mask used to save plm as partial maps"""
        return self.mask_qe !=0

    def p_plm(self, tag=None, seed1=None, seed2=None, cmbset1=None, cmbset2=None, N1=False, stack_type=None, ):
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
        """
        subdir = 'lensrec_N1' if N1 else 'lensrec'
        if self.bundle is not None:
            subdir = f"{subdir}/bundle{self.bundle}"
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
        if self.bundle is not None:
            subdir = f"{subdir}/bundle{self.bundle}"
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

    def p_resp(self, tag):
        """paths to response functions."""
        return self.path(self.outdir, f"respavg_{tag}.npz")

    @staticmethod
    def f_tmp(tag, seed1=None, seed2=None, ktype=None, N1=False, mf_group=0):
        """
        Return file name of a temprary file.

        Parameters
        ----------
        tag: str
            qe or a mvtype, e.g. "TT"/"qMV"/"GMV"/"PP"
        seed1, seed2: int
        ktype: str
            2-letter string
        N1: bool=False
            Indicator for N1-type maps.
        """
        return os.path.join(f"kmap_{tag}_{seed1}_{seed2}_mfgroup{mf_group}_{ktype}{'_N1' if N1 else ''}.tmp")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default=None, type=str, help='path to config file', required=True)
    p.add_argument('-f', '--field', default=None, type=str, help='SPT field')
    p.add_argument('-b', '--bundle', default=None, type=int, help='Bundle id')
    p.add_argument('-n1', action='store_true', help='do N1-type operations')
    p.add_argument('-rdn0', action='store_true', help='do RDN0-type operations')
    return p
