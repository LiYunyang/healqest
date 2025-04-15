import sys
from dataclasses import dataclass
import os
import numpy as np
import yaml
from healqest import utils
from importlib import resources
import healpy as hp
from healqest.ducc_sht import Geometry
from functools import cached_property
import shutil

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0


class Config:
    outdir: str  # output directory
    recdir: str = None # output directory for lensing rec. Default to outdir
    mfsplit: bool

    nside: int # used for ducc wrapper

    def __init__(self, **kwargs):
        # TODO: this should really be a dataclass with all possible arguments
        # specified, typed and with defaults.
        self.__dict__.update(kwargs)

        self.lmin = self['lensing']['lmin']
        self.lmaxT = self['lensing']['lmaxT']
        self.lmaxP = self['lensing']['lmaxP']
        self.Lmax = self['lensing']['Lmax']
        self.lmaxTP = max(self.lmaxT, self.lmaxP)
        self._dict_cls = None
        if self.recdir is None:
            self.recdir = self.outdir

    @classmethod
    def from_yaml(cls, fname, pipeline=False):
        params = yaml.safe_load(open(fname, "r"))
        obj = cls(**params)
        # copy the config file
        if pipeline and rank == 0:
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
        *args : str
            Additional path components to be joined to the path.
        **kwargs : dict
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

        if self.sim_mask is not None:
            # nside = hp.get_nside(self.sim_mask)
            # alm = hp.map2alm(hp.alm2map(alm, nside=nside, )*self.sim_mask, iter=0)
            alm = self._mask_alm(alm, self.sim_mask)
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

    @cached_property
    def sim_mask(self):
        if self['lensing'].get('sim_mask', None):
            return hp.read_map(self.file(self['lensing'].get('sim_mask')))
        else:
            return None

    def p_plm(self, qe, seed1=None, seed2=None, cmbset1=None, cmbset2=None, N1=False, stack_type=None):
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
        if not stack_type:
            fname = f'plm_{qe.upper()}_{seed1}{cmbset1}_{seed2}{cmbset2}.npz'
        else:
            subdir = f"{subdir}/stack"
            fname = f'plmstack_{qe.upper()}_{stack_type}.npz'
        out = self.file(self.recdir, subdir, fname)
        return out

    def p_cls(self, qe, seed1, seed2, ktype1, ktype2, N1=False, ext='dat'):
        """
        Return paths to power spectra files.
        """
        subdir = 'cls/lensrec_N1' if N1 else 'cls'
        assert set(ktype1) == set(ktype2)
        s1, s2, c1, c2 = self.ktype2ij(ktype1, seed1, seed2)
        tag1 = f"{s1}{c1}_{s2}{c2}"
        s1, s2, c1, c2 = self.ktype2ij(ktype2, seed1, seed2)
        tag2 = f"{s1}{c1}_{s2}{c2}"
        fname = f'clkk_k{qe.upper()}_{tag1}_{tag2}.{ext}'
        out = self.file(self.outdir, subdir, fname)
        return out

    def p_reps(self, qe):
        """
        Return paths to response functions.
        """
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
