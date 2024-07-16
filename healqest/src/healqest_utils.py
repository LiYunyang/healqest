import os,sys,git,uuid
import numpy as np
import healpy as hp
from pathlib import Path
import yaml,pickle
import logging as lg

def rebincl(ell,cl, bb):
    #bb   = np.linspace(minell,maxell,Nbins+1)
    Nbins=len(bb)-1
    ll   = (bb[:-1]).astype(np.int_)
    uu   = (bb[1:]).astype(np.int_)
    ret  = np.zeros(Nbins)
    retl = np.zeros(Nbins)
    err  = np.zeros(Nbins)
    for i in range(0,Nbins):
        ret[i]  = np.mean(cl[ll[i]:uu[i]])
        retl[i] = np.mean(ell[ll[i]:uu[i]])
        err[i]  = np.std(cl[ll[i]:uu[i]])
    return ret

def extract_patch(mask,patch):

    nside  = hp.npix2nside(mask.shape[0])
    pix0    = np.where(mask>0.0)[0]

    pixs={}
    pixs[1] = pix0

    # 501-750
    tht,phi   = hp.pix2ang(nside,pix0)
    tht2,phi2 = tht,phi+np.pi
    pixs[3]   = hp.ang2pix(nside,tht2,phi2)

    # 750-1000
    ra,dec    = tp2rd(tht,phi)
    tht3,phi3 = rd2tp(ra,-1*dec)
    pixs[4]   = hp.ang2pix(nside,tht3,phi3)

    # 250-500
    ra,dec    = tp2rd(tht2,phi2)
    tht4,phi4 = rd2tp(ra,-1*dec)
    pixs[2]   = hp.ang2pix(nside,tht4,phi4)

    pidx = pixs[patch]

    return pix0,pidx

def rd2tp(ra,dec):
    """
    Convert ra,dec -> tht,phi
    """
    tht = (-dec+90.0)/180.0*np.pi
    phi = ra/180.0*np.pi
    return tht,phi

def tp2rd(tht,phi):
    """
    Convert tht,phi -> ra,dec
    """
    ra  = phi/np.pi*180.0
    dec = -1*(tht/np.pi*180.0-90.0)
    return ra,dec

def get_mmask(lmax,mmin,verbose=True):
    if verbose:
        print("max=%d mmin=%d"%(lmax,mmin) )
    ell,emm=hp.Alm.getlm(lmax)
    mm=np.ones_like(ell,dtype=np.complex_ )
    mm[emm<mmin]=0
    return mm

def zeropad(cl):
    """add zeros for L=0,1"""
    cl=np.insert(cl,0,0)
    cl=np.insert(cl,0,0)
    return cl

def parse_yaml(file_yaml):
    '''
    Load all settings in yaml
    '''
    print('Loading lensing config: %s'%file_yaml)
    # Read the yaml file
    dict  = yaml.safe_load(Path(file_yaml).read_text())

    # Check healqest version of commit used
    repo  = git.Repo(dict['base']['dir_healqest'],search_parent_directories=True)
    sha   = repo.head.object.hexsha

    print('healqest commit: %s'%sha)

    # Check reconstruction type and return maps and qe needed.
    rectype  = dict['lensrec']['rectype']
    print('Reconstruction type: %s'%rectype)

    recdict  = {'sqe'  : {'maptype1': 'cmbmv',
                          'maptype2': 'cmbmv',
                          'qes'     : ['TT','EE','TE','TB','EB','ET','BT','BE']
                         },
                'gmv'  : {'maptype1': 'cmbmv',
                          'maptype2': 'cmbmv',
                          'qes'     : ['TT_GMV','EE_GMV','TE_GMV','ET_GMV','TB_GMV','BT_GMV','EB_GMV','BE_GMV']
                         },
                'gmvjtp': {'maptype1': 'cmbmv',
                           'maptype2': 'cmbmv',
                           'qes'     : ['TT','TE','TB','ET','EE','EB','BT','BE']
                         },
                'gmvph': {'maptype1': 'cmbmv',
                          'maptype2': 'cmbmv'
                         },
                'mh'   : {'maptype1': 'cmbynull',
                          'maptype2': 'cmbmv'   ,
                          'qes'     : ['TT']
                         },
                'xilc' : {'maptype1': 'cmbynull',
                          'maptype2': 'cmbcibnull',
                          'qes'     : ['TT']
                         }
                }

    dict['maptype1'] = recdict[rectype]['maptype1']
    dict['maptype2'] = recdict[rectype]['maptype2']
    dict['qes']      = recdict[rectype]['qes']
    dict['dir_out']  = dict['outputs']['dir_out']

    # Read Cls from specified files
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
        nhrs  = record.relativeCreated//(1000*60*60)
        nmins = record.relativeCreated//(1000*60)-nhrs*60
        nsecs = record.relativeCreated//(1000)-nmins*60
        record.relativeCreated = "%02d:%02d:%02d"%(nhrs,nmins,nsecs)#, record.relativeCreated//(1000) )
        #print( dtype(record.relativeCreated//(1000)) )
        return super(RelativeSeconds, self).format(record)

def setup_logger():
    print("Setting up logging")
    lg.basicConfig(level = lg.WARNING)
    formatter = RelativeSeconds("[%(relativeCreated)s]  %(message)s")
    lg.root.handlers[0].setFormatter(formatter)


def reduce_lmax(alm, lmax=4000):
    """
    Reduce the lmax of input alm
    """
    lmaxin  = hp.Alm.getlmax(alm.shape[0])
    print( "-- Reducing lmax: lmax_in=%g -> lmax_out=%g"%(lmaxin,lmax) )
    ell,emm = hp.Alm.getlm(lmaxin)
    almout  = np.zeros(hp.Alm.getsize(lmax),dtype=np.complex128)
    oldi=0
    oldf=0
    newi=0
    newf=0
    dl = lmaxin-lmax
    for i in range(0,lmax+1):
        oldf=oldi+lmaxin+1-i
        newf=newi+lmax+1-i
        almout[newi:newf]=alm[oldi:oldf-dl]
        oldi=oldf
        newi=newf
    return almout

def get_nside(lmax):
    """calculates the most appropriate nside based on lmax"""
    nside = np.array([8,16,32,64,128,256,512,1024,2048,4096,8192,16384])
    idx   = np.argmin(np.abs(nside-lmax))
    return nside[idx]

def zeropad(cl):
    """add zeros for L=0,1"""
    cl=np.insert(cl,0,0)
    cl=np.insert(cl,0,0)
    return cl

def load_cambcls(file,lmax=2000,dict=False,dls=False):
    d = np.loadtxt(file)
    ell,sltt,slee,slbb,slte = d[:,(0,1,2,3,4)].T

    if dls==False:
        # Removing the ell factors and padding with zeros (since the file starts with l=2)
        sltt=sltt/ell/(ell+1)*2*np.pi; sltt=zeropad(sltt)
        slee=slee/ell/(ell+1)*2*np.pi; slee=zeropad(slee)
        slte=slte/ell/(ell+1)*2*np.pi; slte=zeropad(slte)
        slbb=slbb/ell/(ell+1)*2*np.pi; slbb=zeropad(slbb)
        ell  = np.insert(ell,0,1); ell=np.insert(ell,0,0)
        ell  = ell[:lmax+1]
        sltt = sltt[:lmax+1]
        slee = slee[:lmax+1]
        slbb = slbb[:lmax+1]
        slte = slte[:lmax+1]

    if dict==False:
        return ell,sltt,slee,slbb,slte
    else:
        d={}
        d['tt']=sltt
        d['ee']=slee
        d['bb']=slbb
        d['te']=slte
        return d

def get_lensedcls(file,lmax=2000,dict=False):
    ell,sltt,slee,slbb,slte=np.loadtxt(file,unpack=True)
    # Removing the ell factors and padding with zeros (since the file starts with l=2)
    sltt=sltt/ell/(ell+1)*2*np.pi; sltt=zeropad(sltt)
    slee=slee/ell/(ell+1)*2*np.pi; slee=zeropad(slee)
    slte=slte/ell/(ell+1)*2*np.pi; slte=zeropad(slte)
    slbb=slbb/ell/(ell+1)*2*np.pi; slbb=zeropad(slbb)
    ell=np.insert(ell,0,1); ell=np.insert(ell,0,0)
    ell  = ell[:lmax+1]
    sltt = sltt[:lmax+1]
    slee = slee[:lmax+1]
    slbb = slbb[:lmax+1]
    slte = slte[:lmax+1]
    if dict==False:
        return ell,sltt,slee,slbb,slte
    else:
        d={}
        d['tt']=sltt
        d['ee']=slee
        d['bb']=slbb
        d['te']=slte
        return d

def get_unlensedcls(file,lmax=2000):
    ell,sltt,slee,slbb,slte,slpp,sltp,slep=np.loadtxt(file,unpack=True)
    # Removing the ell factors and padding with zeros (since the file starts with l=2)
    sltt=sltt/ell/(ell+1)*2*np.pi;             sltt=zeropad(sltt)
    slee=slee/ell/(ell+1)*2*np.pi;             slee=zeropad(slee)
    slbb=slbb/ell/(ell+1)*2*np.pi;             slbb=zeropad(slbb)
    slte=slte/ell/(ell+1)*2*np.pi;             slte=zeropad(slte)
    slpp=slpp/ell/ell/(ell+1)/(ell+1)*2*np.pi; slpp=zeropad(slpp)
    sltp=sltp/(ell*(ell+1))**(1.5)*2*np.pi;    sltp=zeropad(sltp)
    slep=slep/(ell*(ell+1))**(1.5)*2*np.pi;    slep=zeropad(slep)
    ell=np.insert(ell,0,1); ell=np.insert(ell,0,0)
    ell  = ell[:lmax+1]
    sltt = sltt[:lmax+1]
    slee = slee[:lmax+1]
    slbb = slbb[:lmax+1]
    slte = slte[:lmax+1]
    slpp = slpp[:lmax+1]
    sltp = sltp[:lmax+1]
    slep = slep[:lmax+1]
    return ell,sltt,slee,slbb,slte,slpp,sltp,slep

'''
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
'''

def add_clsdict(d,key,cltt,clee,clbb,clte=None):
    d[key]  = {}
    d[key]['tt'] = cltt
    d[key]['ee'] = clee
    d[key]['bb'] = clbb

    if clte is not None:
        d[key]['te'] = clte

    return d

#def get_fl(config,use_unlCls=False):
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

#def get_almbar(qetype,alms1,alms2,config,use_unlCls=True):
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

def get_fl(config,mtype,use_unlCls=False):

    sdict  = {'T':'tt', 'E':'ee', 'B':'bb' }

    lmaxTP = config['lmaxTP']
    lmax   = config['lmax%s'%('T' if mtype=='T' else 'P')]
    lmin   = config['lmin']

    if use_unlCls:
        fl = 1.0/(config['cls']['ucmb'][sdict[mtype]][:lmaxTP+1]+config['cls']['totres'][sdict[mtype]][:lmaxTP+1])
    else:
        fl = 1.0/(config['cls']['lcmb'][sdict[mtype]][:lmaxTP+1]+config['cls']['totres'][sdict[mtype]][:lmaxTP+1])
    fl[:lmin] = 0
    if lmax < lmaxTP: fl[lmax+1:] = 0

    return fl

def get_almbar(config,mtype,cmbid,seed,use_unlCls=True):

    hdudict = {_mtype: c+1 for c,_mtype in enumerate(['T','E','B'])}

    lmaxTP  = config['lmaxTP']
    alm     = hp.read_alm(config['iqu']['dir']+config['iqu']['prefix'].format(cmbid=cmbid,seed=seed),hdu=hdudict[mtype])
    alm     = reduce_lmax(alm,lmax=lmaxTP);

    fl      = get_fl(config,mtype,use_unlCls=use_unlCls)

    almbar  = hp.almxfl(alm,fl)

    return almbar,fl

def load_beam():
    file_beam=dir_base+'/beam/saturn.txt'
    tmp  = np.loadtxt(file_beam)
    bl   = {}
    bl[90],bl[150],bl[220] = tmp[:,1], tmp[:,2],  tmp[:,3]
    return bl

def load_tf(d='/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/tf/',fill_value=0,include_beam=True):
    if include_beam:
        print('Including beam in the transfer function')
        bl = load_beam()

    for freqi in (90,150,220):
        print('Loading TF from %s'%d)
        y = np.load(d+'tf2d_%d.npz'%freqi)['tf2d'].real
        y[np.isnan(y)] = fill_value
        y    = reduce_lmax(y,lmax=lmax)
        if include_beam:
            y = hp.almxlf(y,bl[freqi][:lmax+1])
        tf1d = np.sqrt(hp.alm2cl(y))
        tf2d = y.real
    return tf1d, tf2d

def make_2dmask(mmin,lmax=6000):
    '''Generate a 2d almspace mask'''
    ell,emm = hp.Alm.getlm(lmax)
    w       = np.ones_like(ell,dtype=np.complex_)
    w[emm<mmin]=0
    return w
