from dataclasses import dataclass
import os
import numpy as np
import yaml
from healqest import utils
from importlib import resources
import healpy as hp


class Config:

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

    @classmethod
    def from_yaml(cls, fname):
        params = yaml.safe_load(open(fname, "r"))
        return cls(**params)

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
        args : tuple of str
            Additional path components to be joined to the path.
        kwargs : dict
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

    def load_maps(self, cmbid, seed):
        paths = self['lensing'][f'iqu{cmbid}']
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
