import os, sys, yaml
import numpy as np
import healpy as hp
from pathlib import Path
import yaml, pickle
import logging as lg
from tqdm import tqdm

np.seterr(divide="ignore", invalid="ignore")


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
    if 'includes' in data:
        included_file = data['includes']
        if type(included_file) is not list: included_file = [included_file]

        for inc_f in included_file:
            included_file_path = os.path.join(os.path.dirname(file_path), inc_f)

            # Recursively load the included file
            with open(included_file_path, 'r') as included_f:
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
        return retl,ret
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

def load_cambfiles_dict(dict):

    dict['lensrec']['lmax'] = max(dict['lensrec']['lmaxT'],dict['lensrec']['lmaxP'])

    if "cls" in dict:

        if 'file_lcmb' in dict['cls']:
            try:
                f   = dict['cls']['file_lcmb']; print(f)
                ell = np.loadtxt(f, usecols=[0])
                dd  = ell*(ell+1)/2/np.pi
                dict['cls']['lcmb'] = {n: zeropad(np.loadtxt(f, usecols=[c+1])/dd)[:dict['lensrec']['lmax']+1] for c, n in enumerate(['tt','ee','bb','te']) }
                #print("Setting CMB lensed cls")
            except:
                print("Couldn't load lensed CMB cls -- not setting CMB Cls")

        if 'file_ucmb' in dict['cls']:
            try:
                f   = dict['cls']['file_ucmb']
                ell = np.loadtxt(f, usecols=[0])
                dd  = ell*(ell+1)/2/np.pi
                vv  = (ell*(ell+1))**2/2/np.pi
                qq  = (ell*(ell+1))**(1.5)/2/np.pi
                dict['cls']['ucmb']       = {n: zeropad(np.loadtxt(f, usecols=[c+1])/dd)[:dict['lensrec']['lmax']+1]  for c, n in enumerate(['tt','ee','bb','te']) }
                dict['cls']['ucmb']['pp'] = zeropad(np.loadtxt(dict['cls']['file_ucmb'], usecols=[5])/vv)[:dict['lensrec']['lmax']+1]
                dict['cls']['ucmb']['tp'] = zeropad(np.loadtxt(dict['cls']['file_ucmb'], usecols=[6])/qq)[:dict['lensrec']['lmax']+1]
                dict['cls']['ucmb']['ep'] = zeropad(np.loadtxt(dict['cls']['file_ucmb'], usecols=[7])/qq)[:dict['lensrec']['lmax']+1]
                #print("Setting CMB unlensed cls")
            except:
                print("Couldn't load unlensed CMB cls -- not setting CMB Cls")

        if 'file_gcmb' in dict['cls']:
            try:
                #TODO: This will change if we use a different file! Currently configured for Abhi's file
                f   = dict['cls']['file_gcmb']
                ell = np.loadtxt(f, usecols=[0])
                dd  = ell*(ell+1)/2/np.pi
                dict['cls']['gcmb'] = {n: zeropad(np.loadtxt(f, usecols=[c+1])/dd)[:dict['lensrec']['lmax']+1]  for c, n in enumerate(['tt','ee','bb','te']) }
                #print("Setting CMB gradient cls")
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
    lmaxin = hp.Alm.getlmax(alm.shape[0])
    print("-- Reducing lmax: lmax_in=%g -> lmax_out=%g" % (lmaxin, lmax))
    ell, emm = hp.Alm.getlm(lmaxin)
    almout = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    oldi = 0
    oldf = 0
    newi = 0
    newf = 0
    dl = lmaxin - lmax
    for i in range(0, lmax + 1):
        oldf = oldi + lmaxin + 1 - i
        newf = newi + lmax + 1 - i
        almout[newi:newf] = alm[oldi : oldf - dl]
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
    '''
    All aresp are computed using healqest/src/gmv_resp.py via run script
    pipeline/spt3g_20192020/src/compute_gmvresp.py

    aresp_fname: filename of the analytic GMV/SQE-TT response file
    arespss_fname: filename of the analytic src-src response file
    arespse_fname: filename of the analytic src-phi response file
    '''
    if 'GMV' in estname:
        print("loading %s response"%estname)
        dic = {'GMVTTEETE':1, 'GMVTBEB':2, 'GMV':3}
        assert estname != 'GMVTBEB', "zero response to TBEB"

        resp1  = np.load(aresp_fname)[:, dic[estname]]
        resp2  = np.load(arespss_fname)[:,1]  #[:,1] == [:,3]  and [:,2]==0
        resp12 = np.load(arespse_fname)[:,1] #[:,1] == [:,3]  and [:,2]==0
    else:
        print("loading SQE %s response"%estname)
        assert estname == 'TT', "not hardening non-TT SQE phi"

        resp1  = np.load(aresp_fname)
        resp2  = np.load(arespss_fname)
        resp12 = np.load(arespse_fname)

    resp2[resp2==0]=np.inf  #prevent NaNs

    weight = -1 * resp12 / resp2
    resp_tot = resp1 + weight * resp12

    return resp_tot, weight


def harden_est(plm_e, plm_s, weight):
    #return hardened, unnormalized estimator
    Lmax     = hp.Alm.getlmax(len(plm_s))
    resplmax = len(weight)-1
    if Lmax > resplmax: 
        weight_l = np.zeros(Lmax+1)
        weight_l[:resplmax+1] = weight
        weight   = weight_l.copy()
        print("resp lmax: %i; src-lm Lmax: %i"%(resplmax, Lmax))
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


def get_qes(qetype):
    if (
        qetype == "GMV"
        or qetype == "GMVTTEETE"
        or qetype == "GMVTBEB"
        or qetype == "GMVbhTTprf"
        or qetype == "GMVTTEETEbhTTprf"
    ):
        qes = ["TT", "EE", "EB", "TE", "TB", "EB", "TE", "TB"]

    elif qetype == "MV":
        qes = ["TT", "EE", "EB", "TE", "TB", "EB", "TE", "TB"]

    elif qetype == "PP":
        qes = ["EE", "EB", "BE"]

    elif (
        qetype == "TT"
        or qetype == "EE"
        or qetype == "TE"
        or qetype == "EB"
        or qetype == "TB"
        or qetype == "ET"
        or qetype == "BE"
        or qetype == "BT"
    ):
        qes = [qetype]

    elif qetype == "TEET":
        qes = ["TE", "ET"]

    elif qetype == "EBBE":
        qes = ["EB", "BE"]

    elif qetype == "TBBT":
        qes = ["TB", "BT"]

    elif qetype == 'qPP':
        '''x2 on asymmetric terms'''
        qes = ["EE", "EB"]

    elif qetype == 'qMV':
        '''x2 on asymmetric terms'''
        qes = ["TT", "EE", "EB", "TE", "TB"]

    elif qetype == "qTEET":
        qes = ["TE"]

    elif qetype == "qEBBE":
        qes = ["EB"]

    elif qetype == "qTBBT":
        qes = ["TB"]

    return qes


def get_dvec(
    dir,
    bine,
    nsims,
    N0,
    N1,
    RDN0=None,
    SAN0=None,
    ellfac=1,
    ratio=False,
    curl=False,
    bpwf=None,
    qe="gmv",
    didx=0,
    theory=None,
    R=1,
):
    """
    Returns datavector and covariance
    dir   : directory where the simulations are stored
    bine  : bin edges
    nsims : number of simulations
    N0    : Noise spectra
    N1    :
    RDN0:
    """
    if curl:
        spec = "ww"
    else:
        spec = "kk"

    if SAN0 is not None:
        print("Using SAN0 instead of N0")

    lmax = len(np.load(dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, 1, 1, 1, 1))["cls"][:, 1])-1

    l = np.arange(lmax+1)
    arr = np.zeros((len(bine) - 1, nsims))
    farr = np.zeros((lmax+1, nsims))
    xx = 0
    c = 0

    if theory is None:
        with open(
            "/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software_base/spt3g_software_051223/scratch/yomori/midell/sims/lensed_cmb/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_rawCls.pickle",
            "rb",
        ) as handle:
            clsa = pickle.load(handle)
        
        # Convert raw clpp to clkk
        ell = np.arange(lmax+1)
        tlkk = (ell * (ell + 1) / 2) ** 2 * clsa["lens_potential"][:lmax+1, 0]
        tlkk[:2] = np.inf

    else:
        print('Using provided cls')
        assert len(theory) >= lmax+1, 'length of theory must be >= lmax+1'
        tlkk = theory[:lmax+1]

    l = np.arange(lmax+1)
    t = lambda l: (l * (l + 1)) ** 2 / 4
    v = (0.5 * l[:lmax+1] * (l[:lmax+1] + 1)) ** 2
    v[:2] = np.inf

    rl = rebincl(l, l, bine)

    # ----------Simulation part--------------
    for i in tqdm(range(1, nsims + 1)):
        x = np.load(dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i, i, i))[
            "cls"
        ][:lmax+1, 1]

        if SAN0 is not None:
            # If an array of semi-analytic N0 is provided, use that instead
            N0 = SAN0[:, i - 1]

        # Debiased spectra
        debiased = (x[:lmax+1] - N0[:lmax+1] - N1[:lmax+1]) * R

        # Array of residuals
        farr[:, i - 1] = debiased - tlkk[:lmax+1]
        farr[:4, i - 1] = 0

        if ratio:
            # Case when we wan to compute measurement/theory
            # The ratio is computed first (unbinned) and then binned using defined bin edges or bpwf.
            # In this case, sim-based correction terms are set to 0.

            if bpwf is None:
                rl, rcl = rebincl(l[:lmax+1], debiased / tlkk[:lmax+1], bine, return_ell=True)
                corr = np.zeros_like(rl)
            else:
                rcl = (debiased / tlkk[:lmax+1]) @ bpwf
                corr = np.zeros_like(rl)

        else:
            # Fiducial case when we want the actual spectra

            tlkk0 = np.copy(tlkk)
            tlkk0[:2] = 0

            if bpwf is None:
                if ellfac >= 0:
                    rl, rcl = rebincl(l[:lmax+1], l[:lmax+1] ** (ellfac) * debiased, bine, return_ell=True)
                else:
                    rl, rcl = rebincl(l[:lmax+1], debiased / v, bine, return_ell=True)
            else:
                if ellfac >= 0:
                    rcl = (l[:lmax+1] ** (ellfac) * debiased)[:lmax+1] @ bpwf
                if ellfac < 0:
                    rcl = ((debiased)[:lmax+1] / v) @ bpwf

        arr[:, c] = rcl
        c += 1

    sim_mean = np.mean(farr, axis=1)

    # --------------Data part-----------------
    if RDN0 is None:
        RDN0 = N0

    print("Loading data spectra:")
    print(dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, didx, didx, didx))
    x = np.load(
        dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, didx, didx, didx)
    )["cls"][:, 1]

    debiased = (x[:lmax+1] - RDN0[:lmax+1] - N1[:lmax+1]) * R

    if ratio:
        # Ratio against fiducial theory, no correction is applied
        if bpwf is None:
            rl, rdl = rebincl(l[:lmax+1], debiased / tlkk[:lmax+1], bine, return_ell=True)
            rdl_corr = np.copy(rdl)
        else:
            rl = (
                np.repeat(
                    np.arange(lmax+1, dtype=np.float64)[:, np.newaxis],
                    len(bine) - 1,
                    axis=1,
                ).T
                @ bpwf
            )[0]
            rdl = (debiased / tlkk[:lmax+1]) @ bpwf
            rdl_corr = np.copy(rdl)
    else:
        # Measured power spectra
        if bpwf is None:
            if ellfac >= 0:
                rl, rdl = rebincl(l[:lmax+1], l[:lmax+1] ** (ellfac) * (debiased), bine, return_ell=True)
                _, rdl_corr = rebincl(
                    l[:lmax+1], l[:lmax+1] ** (ellfac) * (debiased - sim_mean), bine, return_ell=True
                )

            else:
                rl, rdl = rebincl(l[:lmax+1], (debiased) / v, bine, return_ell=True)
                _, rdl_corr = rebincl(l[:lmax+1], (debiased - sim_mean) / v, bine, return_ell=True)

        else:
            rl = (
                np.repeat(
                    np.arange(lmax+1, dtype=np.float64)[:, np.newaxis],
                    len(bine) - 1,
                    axis=1,
                ).T
                @ bpwf
            )[0]
            if ellfac >= 0:
                rdl = (l[:lmax+1] ** (ellfac) * (debiased))[:lmax+1] @ bpwf
                rdl_corr = (l[:lmax+1] ** (ellfac) * (debiased - sim_mean))[:lmax+1] @ bpwf
            else:
                rdl_corr = ((debiased - np.mean(farr, axis=1))[:lmax+1] / v) @ bpwf
                rdl = ((debiased - 0 * np.mean(farr, axis=1))[:lmax+1] / v) @ bpwf

    return rl, rdl, np.mean(arr, axis=1), np.std(arr, axis=1), arr, rdl_corr


def loadcls(
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
    startidx=1
):
    print(f'starting from index {startidx}')
    if curl:
        spec = "ww"
    else:
        spec = "kk"

    Lmin, Lmax = np.int32(Lmin), np.int32(Lmax)

    if cltype == "dd":
        return np.load(
            dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, didx, didx, didx, didx)
        )["cls"][: lmax + 1, 1]

    elif cltype == "xx":
        xx = 0
        for i in tqdm(range(startidx, nsims + startidx)):
            xx += np.load(
                dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i, i, i)
            )["cls"][: lmax + 1, 1]
        return xx / nsims

    elif cltype == "N0":
        N0 = 0
        for i in tqdm(range(startidx, nsims + startidx)):
            a = np.load(
                dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i + 1, i, i + 1)
            )["cls"][: lmax + 1, 1]
            b = np.load(
                dir + "cl%s_k%s_%da_%da_%da_%da.npz" % (spec, qe, i, i + 1, i + 1, i)
            )["cls"][: lmax + 1, 1]
            N0 += a + b
        return N0 / nsims

    elif cltype == "N0x":
        N0x = 0
        for i in tqdm(range(startidx, nsims + startidx)):
            ayay = np.load(
                dir + "cl%s_k%s_%da_%db_%da_%db.npz" % (spec, qe, i, i+1, i, i+1)
            )["cls"][: lmax + 1, 1]
            ayya = np.load(
                dir + "cl%s_k%s_%da_%db_%db_%da.npz" % (spec, qe, i, i+1, i+1, i)
            )["cls"][: lmax + 1, 1]
            N0x += (ayay + ayya)
        return N0x / nsims

    elif cltype == "N1":
        assert N0 is not None
        N1 = 0
        for i in tqdm(range(startidx, nsims + startidx)):
            abab = np.load(
                dir + "cl%s_k%s_%da_%db_%da_%db.npz" % (spec, qe, i, i, i, i)
            )["cls"][: lmax + 1, 1]
            abba = np.load(
                dir + "cl%s_k%s_%da_%db_%db_%da.npz" % (spec, qe, i, i, i, i)
            )["cls"][: lmax + 1, 1]
            N1 += (abab + abba) - N0
        return N1 / nsims

    elif cltype == "N1x":
        assert N0 is not None
        NX = 0
        for i in tqdm(range(startidx, nsims + startidx)):
            abab = np.load(
                dir + "cl%s_k%s_%da_%db_%da_%db.npz" % (spec, qe, i, i, i, i)
            )["cls"][: lmax + 1, 1]
            abba = np.load(
                dir + "cl%s_k%s_%da_%db_%db_%da.npz" % (spec, qe, i, i, i, i)
            )["cls"][: lmax + 1, 1]
            NX += abab + abba  # -N0
        return NX / nsims

    elif cltype == "abab":
        assert N0 is not None
        NX = 0
        for i in tqdm(range(startidx, nsims + startidx)):
            abab = np.load(
                dir + "cl%s_k%s_%da_%db_%da_%db.npz" % (spec, qe, i, i, i, i)
            )["cls"][: lmax + 1, 1]
            # abba = np.load(dir+'cl%s_k%s_%da_%db_%db_%da.npz'%(spec,qe,i,i,i,i))['cls'][:lmax+1,1]
            NX += abab  # -N0
        return NX / nsims

    elif cltype == "abba":
        assert N0 is not None
        NX = 0
        for i in tqdm(range(startidx, nsims + startidx)):
            # abab = np.load(dir+'cl%s_k%s_%da_%db_%da_%db.npz'%(spec,qe,i,i,i,i))['cls'][:lmax+1,1]
            abba = np.load(
                dir + "cl%s_k%s_%da_%db_%db_%da.npz" % (spec, qe, i, i, i, i)
            )["cls"][: lmax + 1, 1]
            NX += abba  # -N0
        return NX / nsims

    elif cltype == "RDN0":
        assert N0 is not None
        RDN0 = 0
        for i in tqdm(range(startidx, nsims + startidx)):
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
        return RDN0 / nsims

    elif cltype == "SAN0c":
        didx = 0

        if qe == "tt":
            aresp = np.load(dir + "/..//SAN0/aresp_%s.npy" % qe.upper())

        if qe == "pp":
            aresp = 0
            qes = ["ee", "eb", "eb"]
            for qei in qes:
                aresp += np.load(dir + "/..//SAN0/aresp_%s.npy" % qei.upper())

        l = np.arange(len(aresp))
        v = (0.5 * (l * (l + 1))) ** 2
        ret = np.zeros((4101, nsims))
        for i in tqdm(range(1, nsims + 1)):
            clqq = np.load(dir + "/../SAN0/clqq_%s_curl_%d.npy" % (qe.upper(), i))
            ret[:, i - 1] = v * clqq / aresp**2 * SAN0tf

        print(dir + "/../SAN0/clqq_%s_curl_%d.npy" % (qe.upper(), i))
        return ret  # *lmask

    elif cltype == "SAN0g":
        """ SAN0 for the gradient part """

        assert N0 is not None

        lmax = len(N0) - 1

        dirR = str(Path(dir).parent)

        aresp = 0
        if qe == "tt":
            qes = ["tt"]
        elif qe == "pp":
            qes = ["ee", "eb", "be"]

        for qei in qes:
            aresp = np.load(
                dirR + "_noinpaint/%s/respavg%s.npz" % (qei.upper(), qei.upper())
            )["resp"][: lmax + 1]

        aresp[:10] = np.inf
        aresp[-100:] = np.inf
        aresp[aresp == 0] = np.inf

        l = np.arange(lmax + 1)
        v = (0.5 * (l * (l + 1))) ** 2

        ret = np.zeros((lmax + 1, nsims))

        for i in tqdm(range(1, nsims + 1)):
            ret[:, i - 1] = (
                v
                * np.load(dir + "/../SAN0/clqq_%s_grad_%d.npy" % (qe.upper(), i))[
                    : lmax + 1
                ]
                / aresp**2
                * SAN0tf[: lmax + 1]
            )

        return ret


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
    v = (0.5 * l * (l + 1)) ** 2

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

    return np.nan_to_num(SAN0arr, nan=0)


def get_bpwf(dir_cls, bine, nsims, N0, N1, ellfac=1, curl=False, qe="gmv", lmax=4000):
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

    for i in tqdm(range(1, nsims + 1)):
        x = np.load(dir_cls + f"cl{spec}_k{qe}_{i}a_{i}a_{i}a_{i}a.npz")["cls"][: lmax + 1, 1]
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
