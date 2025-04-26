import sys
import argparse
import os
from typing import Union, get_type_hints

import numpy as np
import yaml
from healqest import utils
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
    outdir: str  # output directory
    recdir: str = None  # output directory for lensing rec. Default to outdir
    mfsplit: bool = True
    field: str = None  # SPT field name, used for output/fname parsing and mask selection
    bundle: int = None  # bundle number, used for output/fname parsing
    """
    the SPT field identifier. This is used to format `outdir`/`recdir`/`dec_range` and mask files.
    """
    nside: int  # used for ducc wrapper

    fmask_qe: Union[str, list[str]]=None  # path(s) to mask used for lensrec
    fmask_ps: Union[str, list[str]]=None  # path(s) to mask used for clpp
    fmask_resp: Union[str, list[str]] = None  # path(s) to mask used for MC resp
    dec_range: Union[list, dict]=None
    spice_kwargs: dict=None  # polspice settings

    save_as_map: bool=False # save plm as map, otherwise as alm.
    lensing: dict  # lumped parameters related to lensing. TODO: re-organize this.

    def __init__(self, **kwargs):
        # TODO: this should really be a dataclass with all possible arguments
        # specified, typed and with defaults.
        self._warned_keys = set()
        self._validate_config(kwargs)
        self.__dict__.update(kwargs)

        self.lmin = self['lensing']['lmin']
        self.lmaxT = self['lensing']['lmaxT']
        self.lmaxP = self['lensing']['lmaxP']
        self.Lmax = self['lensing']['Lmax']
        self.lmaxTP = max(self.lmaxT, self.lmaxP)
        self._dict_cls = None
        if self.recdir is None:
            self.recdir = self.outdir
        if self.field is not None:
            self.outdir = self.outdir.format(field=self.field)
            self.recdir = self.recdir.format(field=self.field)
        if self.bundle is not None:
            self.outdir = self.outdir.format(bundle=self.bundle)
            self.recdir = self.recdir.format(bundle=self.bundle)
        if isinstance(self.dec_range, dict):
            if self.field is not None:
                self.dec_range = self.dec_range[self.field]

        # auto adjust spice kwargs
        if self.spice_kwargs:
            for key in ['apodizesigma', 'thetamax']:
                if self.spice_kwargs.get(key, None) == 'dec':
                    self.spice_kwargs[key] = max(self.dec_range)-min(self.dec_range)
                elif isinstance(self.spice_kwargs[key], dict):
                    self.spice_kwargs[key] = self.spice_kwargs[key][self.field]

    def _validate_config(self, config_dict: dict):
        expected_keys = get_type_hints(self)
        unexpected_keys = set(config_dict.keys()) - set(expected_keys.keys())

        for key in unexpected_keys:
            if key not in self._warned_keys:
                warnings.warn(f"Unexpected config parameter: '{key}'. ", UserWarning, stacklevel=2)
                self._warned_keys.add(key)

    @classmethod
    def from_yaml(cls, fname, field=None, bundle=None):
        params = yaml.safe_load(open(fname, "r"))
        return cls(**params, field=field, bundle=bundle)

    @classmethod
    def from_args(cls, args):
        fname = args.config
        obj = cls.from_yaml(fname, field=args.field, bundle=args.bundle)
        if rank == 0:
            os.makedirs(obj.file(obj.outdir), exist_ok=True)
            shutil.copy(fname, obj.file(obj.outdir))
            script = sys._getframe(1).f_globals['__file__']
            shutil.copy(script, obj.file(obj.outdir))
        return obj

    def __getitem__(self, item):
        # so that the object can be accessed like a dictionary (the old way)
        # but maybe we want to make it more directly with "."
        return self.__dict__[item]

    @staticmethod
    def file(path, *args, **kwargs):
        """
        Formatting the files.

        Parameters
        ----------
        path : str
            The path to the file. If it starts with "/", it is considered an absolute path.
            Otherwise, it is considered relative to the data directory ($HEALQEST_DATA_DIR).
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
            out = os.path.join(os.environ["HEALQEST_DATA_DIR"], path)
        if args:
            out = os.path.join(out, *args)
        if kwargs:
            out = out.format(**kwargs)
        if os.path.exists(out):
            pass
        else:
            pass
            # raise FileNotFoundError(out)
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

    def load_maps(self, cmbid, seed, N1=False,):
        assert cmbid in [1, 2]
        paths = self['lensing'][f'iqu'] if not N1 else self['lensing'][f'iqu_N1']
        # loop in case we have multiple components (cmb, foreground, noise etc.)
        if isinstance(paths, str):
            paths = [paths]
        alm = 0
        for p in paths:
            f = self.file(p, cmbid=cmbid, seed=seed)
            try:
                alm += hp.read_alm(f, hdu=[1, 2, 3])
            except (IndexError, AttributeError):
                _alm = hp.read_alm(f)
                alm += np.array([_alm, _alm, _alm])

        if self.mask_qe is not None:
            alm = self._mask_alm(alm, self.mask_qe)
        return alm

    @property
    def dict_cls(self):
        if self._dict_cls is None:
            file_clnoise = self.file(self['lensing']['filter']['cl_noise'])
            file_clfg = self.file(self['lensing']['filter']['cl_fg'])
            cambcls = str(resources.files('healqest') / 'camb' / self['lensing']['cambcls'])

            dict_cls = {}

            ell, sltt, slee, slbb, slte = utils.get_lensedcls(cambcls, lmax=self.lmaxTP)
            dict_cls = utils.add_clsdict(dict_cls, 'cmb', sltt, slee, slbb)

            cls_noise = np.loadtxt(file_clnoise)[:self.lmaxTP + 1, :6]
            cls_totfg = np.loadtxt(file_clfg)[:self.lmaxTP + 1, :6]
            res = cls_totfg + cls_noise
            dict_cls = utils.add_clsdict(dict_cls, 'res', res[:, 1], res[:, 2], res[:, 3])
            self._dict_cls = dict_cls
        return self._dict_cls

    @property
    def dict_lrange(self):
        return dict(lmin=self.lmin, lmaxTP=self.lmaxTP, lmaxT=self.lmaxT, lmaxP=self.lmaxP, Lmax=self.Lmax)

    @cached_property
    def g(self):
        return Geometry(nside=self.nside, dec_range=getattr(self, 'dec_range', None))

    def _load_mask(self, item):
        if isinstance(item, str):
            return hp.read_map(self.file(item, field=self.field))
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

    def p_plm(self, qe=None, seed1=None, seed2=None, cmbset1=None, cmbset2=None, N1=False, stack_type=None, ):
        """
        Return paths to plm(stacked) files.

        Parameters
        ----------
        qe: str
        seed1, seed2: int
        cmbset1, cmbset2: str
            Single letter strings. Accepted values are 'a', 'b', 'x', 'y'.
        N1: bool=False
            Indicate if the target file is for N1 calculation (they live in a separate directory).
        stack_type: str
            Indicate the stacking type for mean-field calculations.
        """
        subdir = 'lensrec_N1' if N1 else 'lensrec'
        suffix = 'fits' if self.save_as_map else 'npz'
        if not stack_type:
            if self.save_as_map:
                # when saving as partial maps, save all QEs together as columns, so QE doesn't appear in fname.
                fname = f'plm_{seed1}{cmbset1}_{seed2}{cmbset2}.{suffix}'
            else:
                fname = f'plm_{qe.upper()}_{seed1}{cmbset1}_{seed2}{cmbset2}.{suffix}'
        else:
            subdir = f"{subdir}/stack"
            fname = f'plmstack_{qe.upper()}_{stack_type}.{suffix}'
        out = self.file(self.recdir, subdir, fname)
        return out

    def p_cls(self, qe, seed1, seed2, ktype1, ktype2, N1=False, ext='dat'):
        """paths to power spectra files."""
        subdir = 'cls/lensrec_N1' if N1 else 'cls'

        s1, s2, c1, c2 = self.ktype2ij(ktype1, seed1, seed2)
        tag1 = f"{s1}{c1}_{s2}{c2}"
        if ktype2 is not None:
            assert set(ktype1) == set(ktype2)
            s1, s2, c1, c2 = self.ktype2ij(ktype2, seed1, seed2)
            tag2 = f"{s1}{c1}_{s2}{c2}"
            fname = f'clkk_k{qe.upper()}_{tag1}_{tag2}.{ext}'
        else:
            fname = f'clkk_k{qe.upper()}_{tag1}_cross.{ext}'
        out = self.file(self.outdir, subdir, fname)
        return out

    def p_resp(self, qe):
        """paths to response functions."""
        return self.file(self.outdir, f"respavg_{qe.upper()}.npz")

    @staticmethod
    def f_tmp(qe, seed1=None, seed2=None, ktype=None, N1=False, mf_group=0):
        """
        Return file name of a temprary file.

        Parameters
        ----------
        qe: str
        seed1, seed2: int
        ktype: str
            2-letter string
        N1: bool=False
            Indicator for N1-type maps.
        """
        return os.path.join(f"kmap_{qe.upper()}_{seed1}_{seed2}_mfgroup{mf_group}_{ktype}{'_N1' if N1 else ''}.tmp")

    def _mask_alm(self, alm, mask):
        return self.g.map2alm(self.g.alm2map(alm)*mask)

    def get_fl(self, qe, cls):
        # TODO: fix me in the future
        fls = utils.get_fl(cls, self.dict_lrange)
        fl1 = fls[list('TEB').index(qe[0].upper())]
        fl2 = fls[list('TEB').index(qe[1].upper())]
        return fl1, fl2


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default=None, type=str, help='path to config file', required=True)
    p.add_argument('-f', '--field', default=None, type=str, help='SPT field')
    p.add_argument('-b', '--bundle', default=None, type=int, help='Bundle id')
    p.add_argument('-i1', default=1, type=int, help='seed start')
    p.add_argument('-i2', default=1, type=int, help='seed stop')
    p.add_argument('-cmbid1', default=1, type=int, help='cmbid1')
    p.add_argument('-cmbid2', default=1, type=int, help='cmbid2')
    return p
