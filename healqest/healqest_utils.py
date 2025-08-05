import os
import sys
from astropy.io import fits
import numpy as np
import healpy as hp
import importlib
from pathlib import Path
import yaml
import logging as lg
from tqdm import tqdm
import tempfile as tf
from typing import Union
import subprocess
import hashlib
import string
logger = lg.getLogger(__name__)

np.seterr(divide="ignore", invalid="ignore")


class EvalFormatter(string.Formatter):
    """
    More capable string formatter that can evaluate expressions.

    Usage:
    >>> template_string = "original_{A}_and_lower_{B.lower()}"
    >>> EvalFormatter().format(template_string, A='A', B='B')
    """
    def get_field(self, field_name, args, kwargs):
        try:
            val = eval(field_name, {}, kwargs)
        except (NameError, AttributeError):
            val = super().get_field(field_name, args, kwargs)
        return val, field_name


def recursive_merge(main_data, included_data):
    """Recursively merge two dictionaries, with values in main_data taking precedence."""
    for key, value in included_data.items():
        if (
            isinstance(value, dict)
            and key in main_data
            and isinstance(main_data[key], dict)
        ):
            # If both main_data and included_data have a dictionary at this key, merge recursively
            recursive_merge(main_data[key], value)
        else:
            # Otherwise, if the key is not present in main_data, add it
            if key not in main_data:
                main_data[key] = value


def load_yaml(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    # Check if there is an `includes` key
    if "includes" in data:
        included_file = data["includes"]
        if type(included_file) is not list:
            included_file = [included_file]

        for inc_f in included_file:
            included_file_path = os.path.join(os.path.dirname(file_path), inc_f)

            # Recursively load the included file
            with open(included_file_path, "r") as included_f:
                included_data = yaml.safe_load(included_f)

            # Recursively merge included data into the main data, preserving main data values
            recursive_merge(data, included_data)

    return data


def setup_logger(savelog=False, file_log="test.log"):
    if savelog == False:
        print("printing to stdout")
        lg.basicConfig(level=lg.WARNING)
        formatter = RelativeSeconds("[%(relativeCreated)s]  %(message)s")
        lg.root.handlers[0].setFormatter(formatter)
    else:
        dir_log = str(Path(file_log).parent)
        Path(dir_log).mkdir(parents=True, exist_ok=True)
        print("saving log to: %s" % dir_log)
        lg.basicConfig(filename=file_log, filemode="w+", level=lg.WARNING)
        formatter = RelativeSeconds("[%(relativeCreated)s]  %(message)s")
        lg.root.handlers[0].setFormatter(formatter)


def parse_dirname(config):
    return config["lensrec"]["dir_out"].format(
        runname=runname,
        rectype=rectype,
        lminT=lminT,
        lminP=lminP,
        lmaxT=lmaxT,
        lmaxP=lmaxP,
        mmin=mmin,
    )


def rebincl(ell, cl, bb, return_ell=False):
    # bb   = np.linspace(minell,maxell,Nbins+1)
    Nbins = len(bb) - 1
    ll = (bb[:-1]).astype(np.int_)
    uu = (bb[1:]).astype(np.int_)
    ret = np.zeros(Nbins)
    retl = np.zeros(Nbins)
    err = np.zeros(Nbins)
    for i in range(0, Nbins):
        ret[i] = np.mean(cl[ll[i] : uu[i]])
        retl[i] = np.mean(ell[ll[i] : uu[i]])
        err[i] = np.std(cl[ll[i] : uu[i]])
    if return_ell:
        return retl, ret
    else:
        return ret


def extract_patch(mask, patch):
    nside = hp.npix2nside(mask.shape[0])
    pix0 = np.where(mask > 0.0)[0]

    pixs = {}
    pixs[1] = pix0

    # 501-750
    tht, phi = hp.pix2ang(nside, pix0)
    tht2, phi2 = tht, phi + np.pi
    pixs[3] = hp.ang2pix(nside, tht2, phi2)

    # 750-1000
    ra, dec = tp2rd(tht, phi)
    tht3, phi3 = rd2tp(ra, -1 * dec)
    pixs[4] = hp.ang2pix(nside, tht3, phi3)

    # 250-500
    ra, dec = tp2rd(tht2, phi2)
    tht4, phi4 = rd2tp(ra, -1 * dec)
    pixs[2] = hp.ang2pix(nside, tht4, phi4)

    pidx = pixs[patch]

    return pix0, pidx


def rd2tp(ra, dec):
    """
    Convert ra,dec -> tht,phi
    """
    tht = (-dec + 90.0) / 180.0 * np.pi
    phi = ra / 180.0 * np.pi
    return tht, phi


def tp2rd(tht, phi):
    """
    Convert tht,phi -> ra,dec
    """
    ra = phi / np.pi * 180.0
    dec = -1 * (tht / np.pi * 180.0 - 90.0)
    return ra, dec


def get_mmask(lmax, mmin, verbose=True):
    if verbose:
        print("max=%d mmin=%d" % (lmax, mmin))
    ell, emm = hp.Alm.getlm(lmax)
    mm = np.ones_like(ell, dtype=np.complex_)
    mm[emm < mmin] = 0
    return mm


def zeropad(cl):
    """add zeros for L=0,1"""
    cl = np.insert(cl, 0, 0)
    cl = np.insert(cl, 0, 0)
    return cl


def parse_yaml(file_yaml):
    """
    Load all settings in yaml
    """
    import git

    print("Loading lensing config: %s" % file_yaml)
    # Read the yaml file
    dict = load_yaml(file_yaml)

    # Check healqest version of commit used
    repo = git.Repo(dict["base"]["dir_healqest"], search_parent_directories=True)
    sha = repo.head.object.hexsha

    print("healqest commit: %s" % sha)

    # Check reconstruction type and return maps and qe needed.
    rectype = dict["lensrec"]["rectype"]
    print("Reconstruction type: %s" % rectype)

    recdict = {
        "sqe": {
            "maptype1": "cmbmv",
            "maptype2": "cmbmv",
            "qes": ["TT", "EE", "TE", "TB", "EB", "ET", "BT", "BE"],
        },
        "gmv": {
            "maptype1": "cmbmv",
            "maptype2": "cmbmv",
            "qes": [
                "TT_GMV",
                "EE_GMV",
                "TE_GMV",
                "ET_GMV",
                "TB_GMV",
                "BT_GMV",
                "EB_GMV",
                "BE_GMV",
            ],
        },
        "gmvjtp": {
            "maptype1": "cmbmv",
            "maptype2": "cmbmv",
            "qes": ["TT", "TE", "TB", "ET", "EE", "EB", "BT", "BE"],
        },
        "gmvjtp_sep": {
            "maptype1": "cmbmv",
            "maptype2": "cmbmv",
            "qes": ["TT", "TE", "TB", "ET", "EE", "EB", "BT", "BE"],
        },
        "gmvjtp_tteete": {
            "maptype1": "cmbmv",
            "maptype2": "cmbmv",
            "qes": ["TT", "TE", "ET", "EE"],
        },
        "gmvjtp_tbeb": {
            "maptype1": "cmbmv",
            "maptype2": "cmbmv",
            "qes": ["TB", "BT", "EB", "BE"],
        },
        "gmvph": {"maptype1": "cmbmv", "maptype2": "cmbmv"},
        "mh": {"maptype1": "cmbynull", "maptype2": "cmbmv", "qes": ["TT"]},
        "xilc": {"maptype1": "cmbynull", "maptype2": "cmbcibnull", "qes": ["TT"]},
    }

    dict["maptype1"] = recdict[rectype]["maptype1"]
    dict["maptype2"] = recdict[rectype]["maptype2"]
    dict["qes"] = recdict[rectype]["qes"]
    dict["dir_out"] = dict["lensrec"]["dir_out"]

    # Read Cls from specified files
    dict["lensrec"]["lmax"] = dict["lensrec"]["Lmax"]#  max(dict["lensrec"]["lmaxT"], dict["lensrec"]["lmaxP"])

    if "cls" in dict:
        if "file_lcmb" in dict["cls"]:
            try:
                f = dict["cls"]["file_lcmb"]
                print(f)
                ell = np.loadtxt(f, usecols=[0])
                dd = ell * (ell + 1) / 2 / np.pi
                dict["cls"]["lcmb"] = {
                    n: zeropad(np.loadtxt(f, usecols=[c + 1]) / dd)[
                        : dict["lensrec"]["lmax"] + 1
                    ]
                    for c, n in enumerate(["tt", "ee", "bb", "te"])
                }
                # print("Setting CMB lensed cls")
            except:
                print("Couldn't load lensed CMB cls -- not setting CMB Cls")

        if "file_ucmb" in dict["cls"]:
            try:
                f = dict["cls"]["file_ucmb"]
                ell = np.loadtxt(f, usecols=[0])
                dd = ell * (ell + 1) / 2 / np.pi
                vv = (ell * (ell + 1)) ** 2 / 2 / np.pi
                qq = (ell * (ell + 1)) ** (1.5) / 2 / np.pi
                dict["cls"]["ucmb"] = {
                    n: zeropad(np.loadtxt(f, usecols=[c + 1]) / dd)[
                        : dict["lensrec"]["lmax"] + 1
                    ]
                    for c, n in enumerate(["tt", "ee", "bb", "te"])
                }
                dict["cls"]["ucmb"]["pp"] = zeropad(
                    np.loadtxt(dict["cls"]["file_ucmb"], usecols=[5]) / vv
                )[: dict["lensrec"]["lmax"] + 1]
                dict["cls"]["ucmb"]["tp"] = zeropad(
                    np.loadtxt(dict["cls"]["file_ucmb"], usecols=[6]) / qq
                )[: dict["lensrec"]["lmax"] + 1]
                dict["cls"]["ucmb"]["ep"] = zeropad(
                    np.loadtxt(dict["cls"]["file_ucmb"], usecols=[7]) / qq
                )[: dict["lensrec"]["lmax"] + 1]
                # print("Setting CMB unlensed cls")
            except:
                print("Couldn't load unlensed CMB cls -- not setting CMB Cls")

        if "file_gcmb" in dict["cls"]:
            try:
                # TODO: This will change if we use a different file! Currently configured for Abhi's file
                f = dict["cls"]["file_gcmb"]
                ell = np.loadtxt(f, usecols=[0])
                dd = ell * (ell + 1) / 2 / np.pi
                dict["cls"]["gcmb"] = {
                    n: zeropad(np.loadtxt(f, usecols=[c + 1]) / dd)[
                        : dict["lensrec"]["lmax"] + 1
                    ]
                    for c, n in enumerate(["tt", "ee", "bb", "te"])
                }
                # print("Setting CMB gradient cls")
            except:
                print("Couldn't load gradient CMB cls -- not setting CMB Cls")

    return dict


def load_cambfiles_dict(dict):
    dict["lensrec"]["lmax"] = max(dict["lensrec"]["lmaxT"], dict["lensrec"]["lmaxP"])

    if "cls" in dict:
        if "file_lcmb" in dict["cls"]:
            try:
                f = dict["cls"]["file_lcmb"]
                print(f)
                ell = np.loadtxt(f, usecols=[0])
                dd = ell * (ell + 1) / 2 / np.pi
                dict["cls"]["lcmb"] = {
                    n: zeropad(np.loadtxt(f, usecols=[c + 1]) / dd)[
                        : dict["lensrec"]["lmax"] + 1
                    ]
                    for c, n in enumerate(["tt", "ee", "bb", "te"])
                }
                # print("Setting CMB lensed cls")
            except:
                print("Couldn't load lensed CMB cls -- not setting CMB Cls")

        if "file_ucmb" in dict["cls"]:
            try:
                f = dict["cls"]["file_ucmb"]
                ell = np.loadtxt(f, usecols=[0])
                dd = ell * (ell + 1) / 2 / np.pi
                vv = (ell * (ell + 1)) ** 2 / 2 / np.pi
                qq = (ell * (ell + 1)) ** (1.5) / 2 / np.pi
                dict["cls"]["ucmb"] = {
                    n: zeropad(np.loadtxt(f, usecols=[c + 1]) / dd)[
                        : dict["lensrec"]["lmax"] + 1
                    ]
                    for c, n in enumerate(["tt", "ee", "bb", "te"])
                }
                dict["cls"]["ucmb"]["pp"] = zeropad(
                    np.loadtxt(dict["cls"]["file_ucmb"], usecols=[5]) / vv
                )[: dict["lensrec"]["lmax"] + 1]
                dict["cls"]["ucmb"]["tp"] = zeropad(
                    np.loadtxt(dict["cls"]["file_ucmb"], usecols=[6]) / qq
                )[: dict["lensrec"]["lmax"] + 1]
                dict["cls"]["ucmb"]["ep"] = zeropad(
                    np.loadtxt(dict["cls"]["file_ucmb"], usecols=[7]) / qq
                )[: dict["lensrec"]["lmax"] + 1]
                # print("Setting CMB unlensed cls")
            except:
                print("Couldn't load unlensed CMB cls -- not setting CMB Cls")

        if "file_gcmb" in dict["cls"]:
            try:
                # TODO: This will change if we use a different file! Currently configured for Abhi's file
                f = dict["cls"]["file_gcmb"]
                ell = np.loadtxt(f, usecols=[0])
                dd = ell * (ell + 1) / 2 / np.pi
                dict["cls"]["gcmb"] = {
                    n: zeropad(np.loadtxt(f, usecols=[c + 1]) / dd)[
                        : dict["lensrec"]["lmax"] + 1
                    ]
                    for c, n in enumerate(["tt", "ee", "bb", "te"])
                }
                # print("Setting CMB gradient cls")
            except:
                print("Couldn't load gradient CMB cls -- not setting CMB Cls")

    return dict


class RelativeSeconds(lg.Formatter):
    def format(self, record):
        nhrs = record.relativeCreated // (1000 * 60 * 60)
        nmins = record.relativeCreated // (1000 * 60) - nhrs * 60
        nsecs = record.relativeCreated // (1000) - nmins * 60
        record.relativeCreated = "%02d:%02d:%02d" % (
            nhrs,
            nmins,
            nsecs,
        )  # , record.relativeCreated//(1000) )
        # print( dtype(record.relativeCreated//(1000)) )
        return super(RelativeSeconds, self).format(record)


def reduce_lmax(alm, lmax=4000):
    """
    Reduce the lmax of input alm
    """
    lmaxin = hp.Alm.getlmax(alm.shape[-1])
    logger.debug(f"Reducing lmax: lmax_in={lmaxin} -> lmax_out={lmax}")
    almout = np.zeros((*alm.shape[:-1], hp.Alm.getsize(lmax)), dtype=alm.dtype)
    oldi = 0
    newi = 0
    dl = lmaxin - lmax
    for i in range(0, lmax + 1):
        oldf = oldi + lmaxin + 1 - i
        newf = newi + lmax + 1 - i
        almout[..., newi:newf] = alm[..., oldi:oldf - dl]
        oldi = oldf
        newi = newf
    return almout


def get_nside(lmax):
    """calculates the most appropriate nside based on lmax"""
    nside = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    idx = np.argmin(np.abs(nside - lmax))
    return nside[idx]


def zeropad(cl):
    """add zeros for L=0,1"""
    cl = np.insert(cl, 0, 0)
    cl = np.insert(cl, 0, 0)
    return cl


def load_cambcls(file, lmax=2000, dict=False, dls=False):
    d = np.loadtxt(file)
    ell, sltt, slee, slbb, slte = d[:, (0, 1, 2, 3, 4)].T

    if dls == False:
        # Removing the ell factors and padding with zeros (since the file starts with l=2)
        sltt = sltt / ell / (ell + 1) * 2 * np.pi
        sltt = zeropad(sltt)
        slee = slee / ell / (ell + 1) * 2 * np.pi
        slee = zeropad(slee)
        slte = slte / ell / (ell + 1) * 2 * np.pi
        slte = zeropad(slte)
        slbb = slbb / ell / (ell + 1) * 2 * np.pi
        slbb = zeropad(slbb)
        ell = np.insert(ell, 0, 1)
        ell = np.insert(ell, 0, 0)
        ell = ell[: lmax + 1]
        sltt = sltt[: lmax + 1]
        slee = slee[: lmax + 1]
        slbb = slbb[: lmax + 1]
        slte = slte[: lmax + 1]

    if dict == False:
        return ell, sltt, slee, slbb, slte
    else:
        d = {}
        d["tt"] = sltt
        d["ee"] = slee
        d["bb"] = slbb
        d["te"] = slte
        return d


def get_lensedcls(file, lmax=2000, dict=False):
    ell, sltt, slee, slbb, slte = np.loadtxt(file, unpack=True)
    # Removing the ell factors and padding with zeros (since the file starts with l=2)
    sltt = sltt / ell / (ell + 1) * 2 * np.pi
    sltt = zeropad(sltt)
    slee = slee / ell / (ell + 1) * 2 * np.pi
    slee = zeropad(slee)
    slte = slte / ell / (ell + 1) * 2 * np.pi
    slte = zeropad(slte)
    slbb = slbb / ell / (ell + 1) * 2 * np.pi
    slbb = zeropad(slbb)
    ell = np.insert(ell, 0, 1)
    ell = np.insert(ell, 0, 0)
    ell = ell[: lmax + 1]
    sltt = sltt[: lmax + 1]
    slee = slee[: lmax + 1]
    slbb = slbb[: lmax + 1]
    slte = slte[: lmax + 1]
    if dict == False:
        return ell, sltt, slee, slbb, slte
    else:
        d = {}
        d["tt"] = sltt
        d["ee"] = slee
        d["bb"] = slbb
        d["te"] = slte
        return d


def get_unlensedcls(file, lmax=2000):
    ell, sltt, slee, slbb, slte, slpp, sltp, slep = np.loadtxt(file, unpack=True)
    # Removing the ell factors and padding with zeros (since the file starts with l=2)
    sltt = sltt / ell / (ell + 1) * 2 * np.pi
    sltt = zeropad(sltt)
    slee = slee / ell / (ell + 1) * 2 * np.pi
    slee = zeropad(slee)
    slbb = slbb / ell / (ell + 1) * 2 * np.pi
    slbb = zeropad(slbb)
    slte = slte / ell / (ell + 1) * 2 * np.pi
    slte = zeropad(slte)
    slpp = slpp / ell / ell / (ell + 1) / (ell + 1) * 2 * np.pi
    slpp = zeropad(slpp)
    sltp = sltp / (ell * (ell + 1)) ** (1.5) * 2 * np.pi
    sltp = zeropad(sltp)
    slep = slep / (ell * (ell + 1)) ** (1.5) * 2 * np.pi
    slep = zeropad(slep)
    ell = np.insert(ell, 0, 1)
    ell = np.insert(ell, 0, 0)
    ell = ell[: lmax + 1]
    sltt = sltt[: lmax + 1]
    slee = slee[: lmax + 1]
    slbb = slbb[: lmax + 1]
    slte = slte[: lmax + 1]
    slpp = slpp[: lmax + 1]
    sltp = sltp[: lmax + 1]
    slep = slep[: lmax + 1]
    return ell, sltt, slee, slbb, slte, slpp, sltp, slep


"""
def setup_logger(nolog,file_log='test.log'):

    dir_log = str(Path(file_log).parent)
    Path(dir_log).mkdir(parents=True, exist_ok=True)

    if nolog==True:
        logging.basicConfig(
                            format   = '[%(asctime)s] %(message)s',
                            datefmt  = '%H:%M:%S',
                            level    = logging.WARNING)
    else:
        logging.basicConfig(filename = file_log ,
                            filemode = 'w+',
                            format   = '[%(asctime)s] %(message)s',
                            datefmt  = '%H:%M:%S',
                            level    = logging.WARNING)
"""


def add_clsdict(d, key, cltt, clee, clbb, clte=None):
    d[key] = {}
    d[key]["tt"] = cltt
    d[key]["ee"] = clee
    d[key]["bb"] = clbb

    if clte is not None:
        d[key]["te"] = clte

    return d


# def get_fl(config,use_unlCls=False):
#    lmaxTP = config['lmaxTP']
#    if use_unlCls:
#        flT = 1.0/(config['cls']['ucmb']['tt'][:lmaxTP+1]+config['cls']['totres']['tt'][:lmaxTP+1])
#        flE = 1.0/(config['cls']['ucmb']['ee'][:lmaxTP+1]+config['cls']['totres']['ee'][:lmaxTP+1])
#        flB = 1.0/(config['cls']['ucmb']['bb'][:lmaxTP+1]+config['cls']['totres']['bb'][:lmaxTP+1])
#    else:
#        flT = 1.0/(config['cls']['lcmb']['tt'][:lmaxTP+1]+config['cls']['totres']['tt'][:lmaxTP+1])
#        flE = 1.0/(config['cls']['lcmb']['ee'][:lmaxTP+1]+config['cls']['totres']['ee'][:lmaxTP+1])
#        flB = 1.0/(config['cls']['lcmb']['bb'][:lmaxTP+1]+config['cls']['totres']['bb'][:lmaxTP+1])
#    flT[lmaxT+1:] = 0; flT[:lmin] = 0
#    flE[lmaxP+1:] = 0; flE[:lmin] = 0
#    flB[lmaxP+1:] = 0; flB[:lmin] = 0
#    return flT, flE, flB

# def get_almbar(qetype,alms1,alms2,config,use_unlCls=True):
#    if alms1.ndim == 1:
#        tlm1,tlm2 = alms1,alms2
#    else:
#        tlm1,elm1,blm1 = alms1[0],alms1[1],alms1[2]
#        tlm2,elm2,blm2 = alms2[0],alms2[1],alms2[2]

#    flT,flE,flB = get_fl(config,use_unlCls=use_unlCls)

#    lmaxTP = config['lmaxTP']
#    print('Preparing input almbars')
#    if qetype[0]=='T': tlm1 = reduce_lmax(tlm1,lmax=lmaxTP); almbar1 = hp.almxfl(tlm1,flT); flm1= flT
#    if qetype[0]=='E': elm1 = reduce_lmax(elm1,lmax=lmaxTP); almbar1 = hp.almxfl(elm1,flE); flm1= flE
#    if qetype[0]=='B': blm1 = reduce_lmax(blm1,lmax=lmaxTP); almbar1 = hp.almxfl(blm1,flB); flm1= flB
#    if qetype[1]=='T': tlm2 = reduce_lmax(tlm2,lmax=lmaxTP); almbar2 = hp.almxfl(tlm2,flT); flm2= flT
#    if qetype[1]=='E': elm2 = reduce_lmax(elm2,lmax=lmaxTP); almbar2 = hp.almxfl(elm2,flE); flm2= flE
#    if qetype[1]=='B': blm2 = reduce_lmax(blm2,lmax=lmaxTP); almbar2 = hp.almxfl(blm2,flB); flm2= flB
#    return almbar1,almbar2,flm1,flm2


def get_totalcls(cls, lmaxT, lmaxP, lmaxTP, lminT, lminP):
    # return total Cls (signal+fg+noise) for inv-var filtering
    # enters GMV in various places
    totalcls = {}
    totalcls["tt"] = cls["lcmb"]["tt"][: lmaxTP + 1] + cls["res"]["tt"][: lmaxTP + 1]
    totalcls["ee"] = cls["lcmb"]["ee"][: lmaxTP + 1] + cls["res"]["ee"][: lmaxTP + 1]
    totalcls["te"] = cls["lcmb"]["te"][: lmaxTP + 1] + cls["res"]["te"][: lmaxTP + 1]
    totalcls["bb"] = cls["lcmb"]["bb"][: lmaxTP + 1] + cls["res"]["bb"][: lmaxTP + 1]

    bignumber = 1e10
    totalcls["tt"][lmaxT + 1 :] = bignumber
    totalcls["te"][lmaxT + 1 :] = bignumber
    totalcls["ee"][lmaxP + 1 :] = bignumber
    totalcls["bb"][lmaxP + 1 :] = bignumber

    totalcls["tt"][:lminT] = bignumber
    totalcls["te"][:lminT] = bignumber
    totalcls["ee"][:lminP] = bignumber
    totalcls["bb"][:lminP] = bignumber

    return totalcls


def get_aresp_tot(aresp_fname, arespss_fname, arespse_fname, estname):
    """
    All aresp are computed using healqest/src/gmv_resp.py via run script
    pipeline/spt3g_20192020/src/compute_gmvresp.py

    aresp_fname: filename of the analytic GMV/SQE-TT response file
    arespss_fname: filename of the analytic src-src response file
    arespse_fname: filename of the analytic src-phi response file
    """
    if "GMV" in estname:
        print("loading %s response" % estname)
        dic = {"GMVTTEETE": 1, "GMVTBEB": 2, "GMV": 3}
        assert estname != "GMVTBEB", "zero response to TBEB"

        resp1 = np.load(aresp_fname)[:, dic[estname]]
        resp2 = np.load(arespss_fname)[:, 1]  # [:,1] == [:,3]  and [:,2]==0
        resp12 = np.load(arespse_fname)[:, 1]  # [:,1] == [:,3]  and [:,2]==0
    else:
        print("loading SQE %s response" % estname)
        assert estname == "TT", "not hardening non-TT SQE phi"

        resp1 = np.load(aresp_fname)
        resp2 = np.load(arespss_fname)
        resp12 = np.load(arespse_fname)

    resp2[resp2 == 0] = np.inf  # prevent NaNs

    weight = -1 * resp12 / resp2
    resp_tot = resp1 + weight * resp12

    return resp_tot, weight


def harden_est(plm_e, plm_s, weight):
    # return hardened, unnormalized estimator
    Lmax = hp.Alm.getlmax(len(plm_s))
    resplmax = len(weight) - 1
    if Lmax > resplmax:
        weight_l = np.zeros(Lmax + 1)
        weight_l[: resplmax + 1] = weight
        weight = weight_l.copy()
        print("resp lmax: %i; src-lm Lmax: %i" % (resplmax, Lmax))
        print("zero-pad weight to match src-lm lmax")

    return plm_e + hp.almxfl(plm_s, weight)


def get_fl(config, mtype, use_unlCls=False):
    sdict = {"T": "tt", "E": "ee", "B": "bb"}

    lmaxTP = config["lmaxTP"]
    lmax = config["lmax%s" % ("T" if mtype == "T" else "P")]
    lmin = config["lmin"]

    if use_unlCls:
        fl = 1.0 / (
            config["cls"]["ucmb"][sdict[mtype]][: lmaxTP + 1]
            + config["cls"]["totres"][sdict[mtype]][: lmaxTP + 1]
        )
    else:
        fl = 1.0 / (
            config["cls"]["lcmb"][sdict[mtype]][: lmaxTP + 1]
            + config["cls"]["totres"][sdict[mtype]][: lmaxTP + 1]
        )
    fl[:lmin] = 0
    if lmax < lmaxTP:
        fl[lmax + 1 :] = 0

    return fl


def get_almbar(config, mtype, cmbid, seed, use_unlCls=True):
    hdudict = {_mtype: c + 1 for c, _mtype in enumerate(["T", "E", "B"])}

    lmaxTP = config["lmaxTP"]
    alm = hp.read_alm(
        config["iqu"]["dir"] + config["iqu"]["prefix"].format(cmbid=cmbid, seed=seed),
        hdu=hdudict[mtype],
    )
    alm = reduce_lmax(alm, lmax=lmaxTP)
    fl = get_fl(config, mtype, use_unlCls=use_unlCls)

    almbar = hp.almxfl(alm, fl)

    return almbar, fl


def load_beam():
    file_beam = dir_base + "/beam/saturn.txt"
    tmp = np.loadtxt(file_beam)
    bl = {}
    bl[90], bl[150], bl[220] = tmp[:, 1], tmp[:, 2], tmp[:, 3]
    return bl


def load_tf(
    d="/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/",
    fill_value=0,
    include_beam=True,
):
    if include_beam:
        print("Including beam in the transfer function")
        bl = load_beam()

    for freqi in (90, 150, 220):
        print("Loading TF from %s" % d)
        y = np.load(d + "tf2d_%d.npz" % freqi)["tf2d"].real
        y[np.isnan(y)] = fill_value
        y = reduce_lmax(y, lmax=lmax)
        if include_beam:
            y = hp.almxlf(y, bl[freqi][: lmax + 1])
        tf1d = np.sqrt(hp.alm2cl(y))
        tf2d = y.real
    return tf1d, tf2d


def make_almmask(alm_lmax, mmin=0, lmin=0, lmax=6000):
    """Generate a 2d almspace mask"""
    ell, emm = hp.Alm.getlm(alm_lmax)
    w = np.ones_like(ell, dtype=np.complex128)
    w[emm < mmin] = 0
    w[ell < lmin] = 0
    w[ell > lmax] = 0
    return w


def get_qes(qeset):
    """
    Retrieve the estimators needed to compute a given QE set.

    Parameters
    ----------
    qeset : str
        A string representing the QE set.

    Returns
    -------
    list or None
        A list of estimators neesed.
    """

    single = {"TT", "EE", "TE", "EB", "TB", "ET", "BE", "BT", "TTbhTTprf", "GMVbhTTprf", "GMVTTEETEbhTTprf"}

    composite = {
        "GMV": ["TT", "EE", "EB", "TE", "TB", "EB", "TE", "TB"],
        "GMVTTEETE": ["TT", "EE", "TE", "ET"],
        "GMVTBEB": ["TB", "BT", "EB", "BE"],
        "MV": ["TT", "EE", "EB", "TE", "TB", "BE", "ET", "BT"],
        "qMV": ["TT", "EE", "EB", "TE", "TB"],
        "MVnoTT": ["EE", "EB", "TE", "TB", "EB", "TE", "TB"],
        "qMVnoTT": ["EE", "EB", "TE", "TB"],
        "MVnoEB": ["EE", "TE", "TB", "TE", "TB"],
        "qMVnoEB": ["EE", "TE", "TB"],
        "TTEETE": ["TT", "EE", "TE", "ET"],
        "qTTEETE": ["TT", "EE", "TE"],
        "TBEB": ["TB", "BT", "EB", "BE"],
        "qTBEB": ["TB", "EB"],
        "PP": ["EE", "EB", "BE"],
        "qPP": ["EE", "EB"],
        "TEET": ["TE", "ET"],
        "EBBE": ["EB", "BE"],
        "TBBT": ["TB", "BT"],
        "qTEET": ["TE"],
        "qEBBE": ["EB"],
        "qTBBT": ["TB"],
        "qMVTTbhTTprf": ["TTbhTTprf", "EE", "TE", "TB", "EB"],
        "MVTTbhTTprf": ["TTbhTTprf", "EE", "EB", "TE", "TB", "EB", "TE", "TB"],
    }


    if qeset in composite:
        return composite[qeset]

    # For any qetype that is one of the single entry codes.
    elif qeset in single:
        return [qeset]

    else:
        sys.exit("Undefined qeset")


def get_dvec(
    dir,
    bine,
    nsims,
    qe,
    N0,
    N1,
    RDN0=None,
    SAN0=None,
    ellfac=1,
    ratio=False,
    curl=False,
    bpwf=None,
    didx=0,
    theory=None,
    R=1,
    dd=None,
    startidx=1,
    unl=False,
    lmax=4000,
    use_cache=False,
    verbose=True
):
    """
    Returns data vector and covariance.

    Parameters
    ----------
    dir : str
        Directory where the simulations are stored.
    bine : array_like
        Bin edges.
    nsims : int
        Number of simulations.
    qe : str
        QE to use.
    N0 : array_like
        N0 spectra.
    N1 : array_like
        N1 spectra.
    RDN0 : array_like, optional
        RDN0 spectra. Default is None.
    SAN0 : array_like, optional
        Semi-analytic N0 spectra array. Default is None.
    ellfac : int, optional
        Scale by ell factor. Default is 0.
    ratio : bool, optional
        Compute ratio with respect to theory. Default is False.
    curl : bool, optional
        Curl mode flag. Default is False.
    bpwf : array_like, optional
        Band power window function. Default is None.
    didx : int
        Data index.
    theory : array_like, optional
        Theory spectra; only used when ratio is True. Default is None.
    R : float
        Arbitrary scaling factor to multiply.
    dd : array_like, optional
        Data spectra. Default is None.
    startidx : int, optional
        Start index. Default is 1.
    unl : bool, optional
        If True, assume unlensed realization. Default is False.
    lmax : int, optional
        Maximum ell to use. Default is 4000.
    use_cache : bool, optional
        Load from packed file. Default is False.

    Returns
    -------
    dict
        A dictionary containing the following keys:

        'rl' : array_like
            Binned ell
        'rdl' : array_like
            Binned data Cl.
        'rcl' : array_like
            Binned sim mean Cl.
        'err' : array_like
            Standard deviation of the rcl.
        'arr' : array_like
            Full array storing all rcl
        'rdl_corr' : array_like
            Sim mean correct data Cl.
    """
    #sys.stdout = open(os.devnull, 'w')

    spec = "ww" if curl else "kk"

    if SAN0 is not None:
        print("Using SAN0 instead of N0")

    l = np.arange(lmax + 1)
    arr = np.zeros((len(bine) - 1, nsims))
    farr = np.zeros((lmax + 1, nsims))
    xx = 0
    c = 0

    if theory is not None:
        print("Using provided cls")
        assert len(theory) >= lmax + 1, "Length of theory must be >= lmax+1"
        tlkk = theory[: lmax + 1]

    l = np.arange(lmax + 1)
    t = lambda l: (l * (l + 1)) ** 2 / 4
    v = (0.5 * l[: lmax + 1] * (l[: lmax + 1] + 1)) ** 2
    v[:2] = np.inf

    rl = rebincl(l, l, bine)
    
    # ----------Simulation part--------------
    if use_cache:
        if unl:
            # For unlensed, array should be list of N0
            f1 = dir + f"all_cl{spec}_{qe}_xyxy.npy"
            f2 = dir + f"all_cl{spec}_{qe}_xyyx.npy"
            x1 = np.load(f1)
            x2 = np.load(f2)
            uu = x1 + x2
        else:
            xx = np.load(dir + f"all_cl{spec}_{qe}_xxxx.npy")

    for i in tqdm(range(startidx, nsims + startidx), disable=not verbose):
        if unl:
            if use_cache:
                x = uu[: lmax + 1, i - startidx]
            else:
                f1 = dir + f"cl{spec}_k{qe}_{i}a_{i + 1}a_{i}a_{i + 1}a.npz"
                f2 = dir + f"cl{spec}_k{qe}_{i}a_{i + 1}a_{i + 1}a_{i}a.npz"
                x1 = np.load(f1)["cls"][: lmax + 1, 1]
                x2 = np.load(f2)["cls"][: lmax + 1, 1]
                x = x1 + x2
        else:
            if use_cache:
                x = xx[:, i - startidx]
            else:
                f = dir + f"cl{spec}_k{qe}_{i}a_{i}a_{i}a_{i}a.npz"
                x = np.load(f)["cls"][: lmax + 1, 1]

        if SAN0 is not None:
            # If an array of semi-analytic N0 is provided, use that instead
            N0 = SAN0[:, i - startidx]

        # Debiased spectra
        debiased = (x[: lmax + 1] - N0[: lmax + 1] - N1[: lmax + 1]) * R

        # Array of residuals
        farr[:, i - startidx] = debiased - tlkk[: lmax + 1]
        farr[:4, i - startidx] = 0

        if ratio:
            # Case when we wan to compute measurement/theory
            # The ratio is computed first (unbinned) and then binned using defined bin edges or bpwf.
            # In this case, sim-based correction terms are set to 0.

            if bpwf is None:
                rl, rcl = rebincl(
                    l[: lmax + 1], debiased / tlkk[: lmax + 1], bine, return_ell=True
                )
                corr = np.zeros_like(rl)
            else:
                rcl = (debiased / tlkk[: lmax + 1]) @ bpwf
                corr = np.zeros_like(rl)
            #print('ratio')
        else:
            # Fiducial case when we want the actual spectra
            #print('not ratio')
            tlkk0 = np.copy(tlkk)
            tlkk0[:2] = 0

            if bpwf is None:
                if ellfac >= 0:
                    rl, rcl = rebincl(
                        l[: lmax + 1],
                        l[: lmax + 1] ** (ellfac) * debiased,
                        bine,
                        return_ell=True,
                    )
                    #print(ellfac)
                else:
                    rl, rcl = rebincl(
                        l[: lmax + 1], debiased / v, bine, return_ell=True
                    )
            else:
                if ellfac >= 0:
                    rcl = (l[: lmax + 1] ** (ellfac) * debiased)[: lmax + 1] @ bpwf
                    #print(ellfac)
                if ellfac < 0:
                    rcl = ((debiased)[: lmax + 1] / v) @ bpwf

        arr[:, c] = rcl
        c += 1

    sim_mean = np.mean(farr, axis=1)

    # --------------Data part-----------------
    if RDN0 is None:
        RDN0 = N0

    if dd is not None:
        print("Using provided data spectra instead of loading")
        x = dd
    else:
        if use_cache:
            x = np.load(dir + f"all_cl{spec}_{qe}_dddd_didx{didx}.npy")
        else:
            print(
                dir
                + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, didx, didx, didx)
            )
            x = np.load(
                dir
                + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, didx, didx, didx)
            )["cls"][:, 1]

    debiased = (x[: lmax + 1] - RDN0[: lmax + 1] - N1[: lmax + 1]) * R

    if ratio:
        # Ratio against fiducial theory, no correction is applied
        if bpwf is None:
            rl, rdl = rebincl(
                l[: lmax + 1], debiased / tlkk[: lmax + 1], bine, return_ell=True
            )
            rdl_corr = np.copy(rdl)
        else:
            rl = (
                np.repeat(
                    np.arange(lmax + 1, dtype=np.float64)[:, np.newaxis],
                    len(bine) - 1,
                    axis=1,
                ).T
                @ bpwf
            )[0]
            rdl = (debiased / tlkk[: lmax + 1]) @ bpwf
            rdl_corr = np.copy(rdl)
            #print(debiased / tlkk[: lmax + 1])
    else:
        # Measured power spectra
        if bpwf is None:
            if ellfac >= 0:
                rl, rdl = rebincl(
                    l[: lmax + 1],
                    l[: lmax + 1] ** (ellfac) * (debiased),
                    bine,
                    return_ell=True,
                )
                _, rdl_corr = rebincl(
                    l[: lmax + 1],
                    l[: lmax + 1] ** (ellfac) * (debiased - sim_mean),
                    bine,
                    return_ell=True,
                )
                #print(ellfac)

            else:
                rl, rdl = rebincl(l[: lmax + 1], (debiased) / v, bine, return_ell=True)
                _, rdl_corr = rebincl(
                    l[: lmax + 1], (debiased - sim_mean) / v, bine, return_ell=True
                )

        else:
            rl = (
                np.repeat(
                    np.arange(lmax + 1, dtype=np.float64)[:, np.newaxis],
                    len(bine) - 1,
                    axis=1,
                ).T
                @ bpwf
            )[0]
            if ellfac >= 0:
                rdl = (l[: lmax + 1] ** (ellfac) * (debiased))[: lmax + 1] @ bpwf
                rdl_corr = (l[: lmax + 1] ** (ellfac) * (debiased - sim_mean))[
                    : lmax + 1
                ] @ bpwf
                #print(ellfac)
            else:
                rdl_corr = ((debiased - np.mean(farr, axis=1))[: lmax + 1] / v) @ bpwf
                rdl = ((debiased - 0 * np.mean(farr, axis=1))[: lmax + 1] / v) @ bpwf

    #if verbose==False: sys.stdout = sys.stdout
    
    # return rl, rdl, np.mean(arr, axis=1), np.std(arr, axis=1), arr, rdl_corr
    return {
        "rl": rl,
        "rdl": rdl,
        "rcl": np.mean(arr, axis=1),
        "err": np.std(arr, axis=1),
        "arr": arr,
        "rdl_corr": rdl_corr,
    }


def loadcls(
    dir,
    nsims,
    qe,
    cltype,
    N0=None,
    Lmin=0,
    Lmax=4000,
    curl=False,
    R=1,
    #SAN0tf=None,
    lmax=4000,
    didx=0,
    startidx=1,
    use_cache=False,
    verbose=True
):    
    #if verbose==False: sys.stdout = open(os.devnull, 'w')

    if curl:
        spec = "ww"
    else:
        spec = "kk"

    Lmin, Lmax = np.int32(Lmin), np.int32(Lmax)

    if cltype == "dd":
        print("Loading dd", end=" ")

        f1 = dir + f"all_cl{spec}_{qe}_dddd_didx{didx}.npy"

        print(f1)
        if use_cache and os.path.exists(f1):
            print("\033[31mWARNING: Using cached file\033[0m")
            return np.load(f1)
        else:
            print('not loading')
            return np.load(dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, didx, didx, didx))["cls"][: lmax + 1, 1]


    elif cltype == "xx":
        print("Loading xx   [%d->%d] " % (startidx, nsims + startidx - 1), end=" " )

        f1 = dir + f"all_cl{spec}_{qe}_xxxx.npy"
        #print(f1)
        if use_cache and os.path.exists(f1):
            print("\033[31mWARNING: Using cached file\033[0m")
            xx = np.mean(np.load(f1), axis=1)

        else:
            print("\033[31mWARNING: NOT using cached file\033[0m")
            xx = 0
            for i in tqdm(range(startidx, nsims + startidx), disable=not verbose):
                xx += np.load(
                    dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i, i, i)
                )["cls"][: lmax + 1, 1]
            xx = xx / nsims

        return xx

    elif cltype == "uu":
        startidx = 3001
        print("Loading uu   [%d->%d] " % (startidx, nsims + startidx - 1), end=" ")

        f1 = dir + f"all_cl{spec}_{qe}_uuuu.npy"

        if use_cache and os.path.exists(f1):
            print(np.load(f1).shape)
            xx = np.mean(np.load(f1), axis=1)

        else:
            xx = 0
            for i in tqdm(range(startidx, nsims + startidx), disable=not verbose):
                xx += np.load(
                    dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i, i, i)
                )["cls"][: lmax + 1, 1]
            xx = xx / nsims

        return xx

    elif cltype == "N0":
        print("Loading N0   [%d->%d] " % (startidx, nsims + startidx - 1), end=" ")

        f1 = dir + f"all_cl{spec}_{qe}_xyxy.npy"
        f2 = dir + f"all_cl{spec}_{qe}_xyyx.npy"

        if use_cache and os.path.exists(f1) and os.path.exists(f2):
            print("\033[31mWARNING: Using cached file\033[0m")
            x1 = np.load(f1)
            x2 = np.load(f2)
            N0 = np.mean(x1 + x2, axis=1)

        else:
            N0 = 0
            for i in tqdm(range(startidx, nsims + startidx), disable=not verbose):
                a = np.load(
                    dir
                    + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i + 1, i, i + 1)
                )["cls"][: lmax + 1, 1]
                b = np.load(
                    dir
                    + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i + 1, i + 1, i)
                )["cls"][: lmax + 1, 1]
                N0 += a + b
            N0 = N0 / nsims

        return N0

    elif cltype == "N1":
        print("Loading N1   [%d->%d] " % (startidx, nsims + startidx - 1), end=" ")
        f1 = dir + f"all_cl{spec}_{qe}_abab.npy"
        f2 = dir + f"all_cl{spec}_{qe}_abba.npy"

        if use_cache and os.path.exists(f1) and os.path.exists(f2):
            assert N0 is not None
            print("\033[31mWARNING: Using cached file\033[0m")
            x1 = np.load(f1)
            x2 = np.load(f2)
            N1 = np.mean(x1 + x2, axis=1) - N0

        else:
            assert N0 is not None
            N1 = 0
            for i in tqdm(range(startidx, nsims + startidx), disable=not verbose):
                abab = np.load(
                    dir + "cl%s_k%s_%da_%db_%da_%db.npz" % (spec, qe, i, i, i, i)
                )["cls"][: lmax + 1, 1]
                abba = np.load(
                    dir + "cl%s_k%s_%da_%db_%db_%da.npz" % (spec, qe, i, i, i, i)
                )["cls"][: lmax + 1, 1]
                N1 += (abab + abba) - N0
            N1 = N1 / nsims
            np.save(dir + f"N1_{qe}_nsims{nsims}.npy", N1)
        return N1

    elif cltype == "RDN0":
        print("Loading RDN0 [%d->%d] " % (startidx, nsims + startidx - 1), end=" ")

        f1 = dir + f"all_cl{spec}_{qe}_xdxd_didx{didx}.npy"
        f2 = dir + f"all_cl{spec}_{qe}_xddx_didx{didx}.npy"
        f3 = dir + f"all_cl{spec}_{qe}_dxdx_didx{didx}.npy"
        f4 = dir + f"all_cl{spec}_{qe}_dxxd_didx{didx}.npy"

        if (
            use_cache
            and os.path.exists(f1)
            and os.path.exists(f2)
            and os.path.exists(f3)
            and os.path.exists(f4)
        ):
            assert N0 is not None
            print("\033[31mWARNING: Using cached file\033[0m")
            x1 = np.load(f1)
            x2 = np.load(f2)
            x3 = np.load(f3)
            x4 = np.load(f4)
            RDN0 = np.mean(x1 + x2 + x3 + x4, axis=1) - N0

        else:
            assert N0 is not None
            RDN0 = 0
            for i in tqdm(range(startidx, nsims + startidx), disable=not verbose):
                xdxd = np.load(
                    dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, didx, i, didx)
                )["cls"][: lmax + 1, 1]
                xddx = np.load(
                    dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, didx, didx, i)
                )["cls"][: lmax + 1, 1]
                dxdx = np.load(
                    dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, i, didx, i)
                )["cls"][: lmax + 1, 1]
                dxxd = np.load(
                    dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, i, i, didx)
                )["cls"][: lmax + 1, 1]
                RDN0 += (xdxd + xddx + dxdx + dxxd) - N0
            RDN0 = RDN0 / nsims

        #if verbose==False: sys.stdout = sys.stdout

        return RDN0


def loadcls_unlcov(
    dir,
    nsims,
    cltype,
    N0=None,
    Lmin=0,
    Lmax=4000,
    curl=False,
    R=1,
    qe="gmv",
    SAN0tf=None,
    lmax=4000,
    didx=0,
    startidx=1,
):
    print(f"starting from index {startidx}")
    if curl:
        spec = "ww"
    else:
        spec = "kk"

    Lmin, Lmax = np.int32(Lmin), np.int32(Lmax)

    if cltype == "N0":
        N0 = np.zeros((lmax + 1, nsims))
        for i in tqdm(range(startidx, nsims + startidx), disable=not verbose):
            a = np.load(
                dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i + 1, i, i + 1)
            )["cls"][: lmax + 1, 1]
            b = np.load(
                dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i + 1, i + 1, i)
            )["cls"][: lmax + 1, 1]
            N0[:, i - startidx] = a + b
        return N0


def get_SAN0(dir, qe, nsims, N0=None, lmax=4000, mode="grad"):
    """
    Return array of SAN0, optionally calibrating against sim based N0.
    Requires the outputs from compute_semianalyticN0_sqe.py

    Parameters:
        dir (str): Directory of SAN0
        qe (str): tt/ee/eb/be/te/et/mv/gmv
        nsims (int): Number of simulation realizations to use
        N0 (np.ndarray): N0 bias (in Nlkk units)
        lmax (int): lmax
        mode (str): grad or curl

    Returns:
        array: (lmax,nsim) sized array containing SAN0 for each realization.
    """
    qe = qe.upper()

    l = np.arange(lmax + 1)
    v = 1 #(0.5 * l * (l + 1)) ** 2 #SAN0 now in Nlkk units

    if N0 is None:
        # No renormalization, just return ones
        print("N0 was not provided. Not adjusting SAN0")
        TF = np.ones(lmax + 1)
    else:
        # Compute the mean SAN0 and calculate transfer function
        print("N0 was provided. Normalizing SAN0 to the provided N0")
        c = 0
        SAN0 = 0
        for i in range(1, nsims + 1):
            SAN0 += v * np.load(dir + f"/clqq_{qe}_{mode}_{i}.npy")[: lmax + 1]
            c += 1
        SAN0 /= c
        TF = N0 / SAN0

    SAN0arr = np.zeros((lmax + 1, nsims))
    for i in range(1, nsims + 1):
        SAN0arr[:, i - 1] = (
            TF * v * np.load(dir + f"/clqq_{qe}_{mode}_{i}.npy")[: lmax + 1]
        )

    return np.nan_to_num(SAN0arr, nan=0), TF


def get_bpwf(
    dir_cls,
    bine,
    nsims,
    qe,
    N0,
    N1,
    ellfac=1,
    curl=False,
    lmax=4000,
    use_cache=False,
    verbose=True
):
    """
    Return band power window function, based on the scatter measured
    from an ensemble of simulations.

    Parameters:
        dir_cls (str): Directory of cls
        bine (np.ndarray):
        nsims (int): Number of simulation realizations to use
        N0 (np.ndarray): N0 bias (in Nlkk units)
        N1 (np.ndarray): N1 bias (in Nlkk units)
        ellfac (int): Number of ell factors on top of clkk * (ell)^ellfac
        curl (bool): grad/curl
        qe (str): tt/ee/eb/be/te/et/mv/gmv
        lmax (int): lmax

    Returns:
        array: (lmax,nbins) sized array containing bpwf for each bin.
    """

    spec = "ww" if curl else "kk"

    l = np.arange(lmax + 1)
    v = (0.5 * l * (l + 1)) ** 2
    v[0] = np.inf

    # Compute variance for every ell, over nsims
    arr = np.zeros((lmax + 1, nsims))

    f1 = dir_cls + f"all_cl{spec}_{qe}_xxxx.npy"

    if use_cache and os.path.exists(f1):
        xx = np.load(f1)
        for i in tqdm(range(1, nsims + 1), disable=not verbose):
            if ellfac >= 0:
                arr[:, i - 1] = l ** (ellfac) * (
                    xx[:, i - 1] - N0[: lmax + 1] - N1[: lmax + 1]
                )
            else:
                arr[:, i - 1] = (xx[:, i - 1] - N0[: lmax + 1] - N1[: lmax + 1]) / v

    else:
        for i in tqdm(range(1, nsims + 1), disable=not verbose):
            x = np.load(dir_cls + f"cl{spec}_k{qe}_{i}a_{i}a_{i}a_{i}a.npz")["cls"][
                : lmax + 1, 1
            ]
            if ellfac >= 0:
                arr[:, i - 1] = l ** (ellfac) * (x - N0[: lmax + 1] - N1[: lmax + 1])
            else:
                arr[:, i - 1] = (x - N0[: lmax + 1] - N1[: lmax + 1]) / v

    # Compute bandpower window function
    vv = np.std(arr, axis=1)
    bpwf = np.zeros((lmax + 1, len(bine) - 1))

    for i in range(len(bine) - 1):
        bi = np.int32(bine[i])
        bf = np.int32(bine[i + 1])
        vbin = np.sum(1 / vv[bi:bf] ** 2)
        bpwf[bi:bf, i] = (1 / vv[bi:bf] ** 2) / (vbin)

    return bpwf


def kspice(m1: Union[np.ndarray, str, list],
           m2: Union[np.ndarray, str, list] = None,
           weight1: Union[np.ndarray, str] = None,
           weight2: Union[np.ndarray, str] = None,
           *,
           lmax=-1,
           apodizetype=1,
           apodizesigma: Union[float, str] = "NO",
           thetamax:float = 180,
           tolerance:float = 5e-8,
           subav:bool = False,
           subdipole:bool = False,
           script=False,
           cl_out: str = None, spice:str = None, kernel=False):
    """
    A python wrapper for PolSpice for temperature (kappa) file only.

    Notes
    -----
    For ka

    Parameters
    ----------
    m1: np.ndarray(3, npix)
        map1 for PS estimation.
    m2: np.ndarray(3, npix), optional.
        map2 for cross PS estimation. If None, m2=m1. Default: None.
    weight1, weight2: np.ndarray=None.
        shape (npix,). The weight map for map1/2. If None,no weights are applied.
        Note that when the `weight2` is None, if `m2` is specified, `weight2` will be considered as FULL SKY rather
        than the same as `weight1`.
    lmax: int=-1.
        The maximum ell number for PS computation. It is advised set lmax=3*nside-1
        (or lmax=-1) for minimum aliasing.
    apodizetype: int=1.
        The apodization type for angular correlation function apodization.
            - 0: the correlation function is multiplied by a gaussian window
                + equal to 1 at theta=0.
                + equal to 0.5 at theta= -apodizesigma/2.
                + equal to 1/16 at theta= -apodizesigma.
            - 1: the correlation function is multiplied by a cosine window
                + equal to 1 at theta=0.
                + equal to 0.5 at theta= -apodizesigma/2.
                + equal to 0 at theta= -apodizesigma.
    apodizesigma: float or str='NO'.
        scale factor in DEGREES of the correlation function tappering. For better
        results, ``apodizesigma`` should be close to ``thetamax``. Use 'NO' to
        disable apodization.
    thetamax: float (0-180)=180.
        The maximum angular distance (in deg) for computing angular-correlation
        function.
    tolerance: float=5e-8.
        Tolerance for convergence.
    subav: bool=False.
    subdipole: bool=False.
    script: bool=False
        If True, return the command line script to be executed.
    cl_out: str
        If present, the output Cl will be write to this file
    spice: str=None
        Path to spice binary
    kernel: bool=False
        If True, return the mode coupling matrix of shape (lmax+1, 2lmax+1).

    Returns
    -------
    [command]: str
        The command line script to be executed.
    [clhat]: np.ndarray(1, nlmax+1)
        PS in orders of: TT
    [kernel]: np.ndarray
        shape (lmax+1, 2lmax+1)

    Notes
    -----
    The wrapper forces ``decouple`` to be True.

    References
    ----------
    PolSpice: http://www2.iap.fr/users/hivon/software/PolSpice/README.html
    """

    dtype = np.float64

    # locate spice binary
    if spice is None:
        spice_bin = os.environ.get("POLSPICE_BIN", os.path.expanduser("~/.local/bin"))
        spice = os.path.join(spice_bin, f"spice_SP")
        if not os.path.exists(spice):
            spice = os.path.join(spice_bin, f"spice_DP")
        if not os.path.exists(spice):
            spice = os.path.join(spice_bin, f"spice")
    else:
        assert os.path.exists(spice)

    # locate the cached polspice configuration
    polspice_config = os.path.expanduser("~/.local/share/polspice")
    if not os.path.exists(polspice_config):
        os.makedirs(polspice_config, exist_ok=True)

    command = [
        spice,
        "-verbosity", "0",
        "-nlmax", str(lmax),
        "-overwrite", "YES",
        "-polarization", "NO",
        "-pixelfile", "NO",
        "-pixelfile2", "NO",
        "-decouple", "YES",
        "-symmetric_cl", "NO",
        "-tolerance", str(tolerance),
        "-apodizetype", str(apodizetype),
        "-apodizesigma", str(apodizesigma),
        "-thetamax", str(thetamax),
        "-subav", "NO" if not subav else "YES",
        "-subdipole", "NO" if not subdipole else "YES",
        "-corfile", "NO",
        # "-verbosity", "2",
    ]
    if m2 is None and weight2 is not None:
        # normally we don't want to do this
        m2 = m1
    with tf.TemporaryDirectory(prefix='spice', ) as tmp:
        for item, name in zip([m1, weight1, m2, weight2],
                          ['mapfile', 'weightfile', 'mapfile2', 'weightfile2']):
            if item is not None:
                if isinstance(item, str):
                    fname = item
                else:
                    fname = os.path.join(tmp, f"{name}.fits")
                    hp.write_map(fname, item, overwrite=True, dtype=dtype,
                                 partial=True if name.startswith("mapfile") else False)
                command += [f"-{name}", fname]
        if cl_out is None:
            cl_out = os.path.join(tmp, f"cls.dat")
        command += [f"-clfile", cl_out]
        if kernel:
            kernel_out = os.path.join(tmp, f"kernel.dat")
            command += ["-kernelsfileout", kernel_out]
        if script:
            return command
        try:
            result = subprocess.run(command, capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed with exit code {e.returncode}")
            print(f"🔧 Command: {' '.join(e.cmd)}")
            if e.stdout:
                print(f"📤 Stdout:\n{e.stdout.strip()}")
            if e.stderr:
                print(f"📣 Stderr:\n{e.stderr.strip()}")
            sys.exit(e.returncode)

        if kernel:
            return fits.open(kernel_out)[0].data[0, :, :].T
        try:
            ell, *clhat = np.loadtxt(cl_out).T
        except ValueError as e:
            print(result.stdout)
            raise e
        clhat = np.array(clhat[0])
        return clhat


def map_or_alm(m):
    """
    Check if object is a map (True) or an alm object.
    """
    try:
        nside = hp.get_nside(m)
        return True
    except TypeError:
        return False


def kappa_spectrum(m1: Union[np.ndarray, str, list],
                   m2: Union[np.ndarray, str, list] = None,
                   mask1: Union[np.ndarray, str] = None,
                   mask2: Union[np.ndarray, str] = None,
                   mask_alm=True, g=None, anafast=True, nside=None, cl_out:str = None, **kwargs):
    """
    General power spectrum estimator

    Parameters
    ----------
    m1, m2: np.ndarray or str
        1d array of map or file(.fits) name of maps. If `synfast=True`, then m1/m2 can be alm/map array or map
        fname, but if `synfast=False`, then m1/m2 should be map fnames or map arrays.
    mask1, mask2: np.ndarray or str
        mask (binary or float) array or file names.
    mask_alm: bool=True
        If true, assume the input alm is unmasked and apply alm2map->mask->map2alm operations.
    g: Geometry
        ducc wrapper object. `g.nside` attribute should match that of the maps.
    anafast: bool=True
        If True, use hp.alm2cl to perform quick power spectrum estimation. The fsky-correction is automatically
        applied so the returned power spectrum should be unbiased.
    nside: int
        used to convert alm2map if `m1/m2` are alm objects and `g` is not given. This is ignore if `anafast=True`
    cl_out: str=None
        Optional output directory.
    kwargs: dict
        kspice keyword arguments.
    """
    def _alm2alm(obj, mask_obj):
        if isinstance(mask_obj, str):
            mask = hp.read_map(mask_obj)
        else:
            mask = mask_obj
        lmax = None
        if isinstance(obj, str):
            m = hp.read_map(obj)
        else:
            if map_or_alm(obj):
                m = obj
            else:
                lmax = hp.Alm.getlmax(len(obj))
                if mask is not None and mask_alm:
                    if g is None:
                        nside = hp.get_nside(mask)
                        m = hp.alm2map(obj, nside=nside)
                    else:
                        m = g.alm2map(obj, )
                else:
                    return obj, mask
        if m is not None:
            if mask is not None:
                m *= mask
            func = hp.map2alm if g is None else g.map2alm
            return func(m, iter=0, lmax=lmax), mask
        raise ValueError

    if anafast:
        alm1, mask1 = _alm2alm(m1, mask1)
        if m2 is None:
            out = hp.alm2cl(alm1)
        else:
            if mask2 is None:
                mask2 = mask1
            alm2, mask2 = _alm2alm(m2, mask2)
            out = hp.alm2cl(alm1, alm2)
        if mask1 is None:
            fsky = 1
        elif mask2 is None:
            fsky = np.mean(mask1)
        else:
            fsky = np.mean(mask1*mask2)
        out /= fsky
        if cl_out is not None:
            l = np.arange(out.shape[-1])
            np.savetxt(cl_out, np.array([l, out]).T)
        return out
    else:
        data = {'m1': m1, 'm2': m2}
        for key, obj in data.items():
            if obj is not None and not isinstance(obj, str):
                if not map_or_alm(obj):
                    if g is None:
                        data[key] = hp.alm2map(obj, nside=nside)
                    else:
                        data[key] = g.alm2map(obj)
                else:
                    data[key] = np.asarray(obj, dtype=np.float64)
        # if m1/m2 are given as file names, then they are assumed to be maps.
        return kspice(m1=data['m1'], m2=data['m2'], weight1=mask1, weight2=mask2, cl_out=cl_out, **kwargs)


def read_map(fname, field=(0, ), dtype=None, partial=False, hdu=1, h=False, use_hp=False):
    """A wrapper to read the partial maps, as fits or npy files.

    Parameters
    ----------
    fname: str
        a path to '.npy' or '.fits' file.
    field: int/str or list of int/str
        column(s) to read from the FITS file or column index for npy arrays. If now, grab all data columns.
    dtype: str or type
    partial : bool, optional
        If True, fits file is assumed to be a partial-sky file with explicit indexing, and the bad pixels are set
        to hp.UNSEEN. If False, implicit indexing is assumed and bad pixels are set to 0  Default: False.
        A partial sky file is one in which OBJECT=PARTIAL and INDXSCHM=EXPLICIT,
        and the first column is then assumed to contain pixel indices.
        A full sky file is one in which OBJECT=FULLSKY and INDXSCHM=IMPLICIT.
        At least one of these keywords must be set for the indexing
        scheme to be properly identified.
    hdu : int, optional
        the header number to look at (start at 0)
    h : bool, optional
        If True, return also the header. Default: False.
    use_hp: bool=False
        If True, use the healpy read_map to read fits map, otherwise, use the faster IO code.
    """
    if isinstance(field, (str, int)):
        field = [field]

    def _allocate(nside,):
        if partial:
            return np.full((len(field), hp.nside2npix(nside)), hp.UNSEEN, dtype=dtype)
        else:
            return np.zeros((len(field), hp.nside2npix(nside)), dtype=dtype)

    if os.path.splitext(fname)[1] == '.npy':
        """load npy partial maps with index stored in parent directories"""
        idx_dir = fname
        c = 3
        while c > 0:
            idx_dir = os.path.dirname(idx_dir)
            try:
                loaded = np.load(os.path.join(idx_dir, 'index.npz'))
                break
            except FileNotFoundError:
                c -= 1
        else:
            raise FileNotFoundError(f'partial map index file not found recursively under {idx_dir}')
        index = loaded['index']
        m = np.load(fname, mmap_mode='r')
        if field is None:
            field = np.arange(m.shape[0])
        out = _allocate(nside=loaded['nside'])

        for idx, j in enumerate(field):
            out[idx, index] = m[j]
        return np.squeeze(out)
    else:
        """load fits partial maps"""
        from astropy.io import fits
        with fits.open(fname, memmap=True) as hdul:
            names = hdul[hdu].columns.names.copy()
            try:
                # for partial maps, we skip the index column
                names.remove('PIXEL')
            except ValueError:
                pass
            fields_num = []
            fields_name = []
            if field is None:
                field = names
            for c in field:
                if isinstance(c, str):
                    if c in names:
                        fields_num.append(names.index(c))
                        fields_name.append(c)
                    else:
                        raise ValueError(f"Column {c} not found in the FITS file: {names}")
                elif isinstance(c, (int, np.integer)):
                    fields_num.append(c)
                    fields_name.append(names[c])
                else:
                    raise TypeError(f"field {c} ({type(c)})?")
            if use_hp:
                return hp.read_map(fname, field=tuple(fields_num), dtype=dtype, hdu=hdu, h=h, partial=partial)
            else:
                out = _allocate(nside=int(dict(hdul[hdu].header)['NSIDE']))
                for j, name in enumerate(fields_name):
                    out[j, hdul[hdu].data['PIXEL']] = hdul[hdu].data[name]
                if h:
                    return np.squeeze(out), hdul[hdu].header
                else:
                    return np.squeeze(out)


def generate_seed(seed, cmbset, bundle=None, extra_tag=None):
    """Generate random seed."""
    return int(hashlib.sha256(f"{cmbset}/{seed}/{bundle}/{extra_tag}".encode()).hexdigest()[:8], base=16)


def cinv_io(fname, maps=None, fl=None, eps=None, return_eps=False):
    """
    Read and write cinv maps.

    Parameters
    ----------
    fname : str
        File name.
    maps: array=None
        shape (1, npix) or (3, npix) map. If None, read and return the maps.
    fl: array=None
        shape (1, lmax+1) or (3, lmax+1) for QE weights flT, flE, flB. If None, read and return the weights.
    eps: array= None
        1d array containing the convergence chain of the cinv run.
    return_eps: bool=False
        If True, return the eps chain of the cinv file

    Returns
    -------
    maps: array
        shape (1, npix) or (3, npix) map
    fl: array
        shape (1, lmax+1) or (3, lmax+1) for QE weights flT, flE, flB
    """
    if maps is None:
        if return_eps:
            hdu = fits.open(fname)[3]
            return hdu.data['eps']
        maps = read_map(fname, field=None, partial=False, hdu=1, dtype=np.float64)
        hdu = fits.open(fname)[2]
        fl = np.array([hdu.data[_.name] for _ in hdu.columns])
        return np.atleast_2d(maps), fl
    else:
        assert len(maps) in (1, 3)
        assert len(fl) in (1, 3)
        hp.write_map(fname, maps, overwrite=True, dtype=np.float64, partial=True)
        with fits.open(fname, mode='update') as hdul:
            hdul.append(fits.BinTableHDU.from_columns([
                fits.Column(name=f"fl{'teb'[i]}", array=_fl, format='D') for i, _fl in enumerate(fl)]))
            if eps is not None:
                hdul.append(fits.BinTableHDU.from_columns([fits.Column(name="eps", array=eps, format='D')]))
            hdul.flush()


def get_spice_kernel(nside, lmax, thetamax=None, apodizesigma=None, apodizetype=None):
    """Return the polspice coupling kernel of shape (lmax+1, 2lmax+1)"""

    if thetamax is None:
        thetamax = kspice.__kwdefaults__['thetamax']
    if apodizesigma is None:
        apodizesigma = kspice.__kwdefaults__['apodizesigma']
    if apodizetype is None:
        apodizetype = kspice.__kwdefaults__['apodizetype']
    fname = f"n{nside}_lmax{lmax}_thetamax{thetamax}_apodizesigma{apodizesigma}_apodizetype{apodizetype}.npy"
    cache_dir = os.environ.get("HEALQEST_IO_ROOT")
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~/.local/share"), "healqest")
    else:
        cache_dir = os.path.join(cache_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    path = os.path.join(cache_dir, fname)
    if os.path.exists(path):
        return np.load(path)
    else:
        logger.warning(f"cache {fname} not found, computing it...")
        zero = np.random.normal(0, 1, hp.nside2npix(nside))
        K = kspice(m1=zero, lmax=lmax, thetamax=thetamax, apodizesigma=apodizesigma, apodizetype=apodizetype,
                   kernel=True)
        logger.info(f"cache saved to {path}")
        np.save(path, K)
        return K


def load_module(module_name, file_path):
    """load a module from a given file path.

    Parameters
    ----------
    module_name: str
        A designated name for the module (used only internally for namespace consistency). If the name starts with
        "healqest.", then the logging level is set properly as the rest of the healqest modules.
    file_path: str
        Path to the Python file containing the module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_diag_almbar(s, alms, config, cls, nlres):
    ss = f"{s.lower()}{s.lower()}"
    assert ss in ['tt', 'ee', 'bb']
    lmax = config.lmaxT if ss == 'tt' else config.lmaxP
    lmin = config.lminT if ss=='tt' else config.lminP
    fl = 1.0 / (cls[ss][:config.lmax + 1] + nlres[ss][:config.lmax + 1])
    fl[:lmin] = 0
    fl[lmax + 1:] = 0
    if alms is None:
        return fl
    alm = reduce_lmax(alms['teb'.index(s.lower())], config.lmax)
    alm = hp.almxfl(alm, fl)
    return alm, fl


def dec2tf2d(lx, dec1, dec2):
    """Compute the tf2d (alm) boundary parameters given the SPT field.

    For a given lx cut in a field between dec1 and dec2, the accessible alm space
    is defined by a trapezoid with following exclusions: l < lx; m < m1, and m > k*l.
    This function computes m1 and k (also returns lx) for the trapezoid.

    Parameters
    ----------
    lx: int
        The cut-off multipole of the time-domian filter
    dec1: float
        The "bottom" declination range (higher absolute value)
    dec2: float
        The "top" declination range (higher absolute value)

    Returns
    -------
    lx: int
        l below lx should be 0
    m1: int
        m below m should be 0
    k: float
        m above k* l should be 0. This value only depend on `dec2` and is useful in computing the effective transfer
        function given some m-cut: tf1d = np.sqrt(hp.alm2cl(tf2d)/k)
    """
    assert 0>dec2>dec1, "dec1 and dec2 should be negative for SPT fields!"
    m1 = int(np.floor(lx * np.sin(np.pi / 2 - np.deg2rad(-dec1))))
    k = np.sin(np.pi / 2 - np.deg2rad(-dec2))
    return lx, m1, k
